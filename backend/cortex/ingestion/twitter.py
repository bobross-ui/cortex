from collections.abc import Iterator
from datetime import datetime, timezone
import html
from pathlib import Path
import time
from cortex.ingestion.base import SourceParser
from cortex.ingestion.normalize import iter_js_array, load_js_array, normalize_tweet, parse_twitter_dt
from cortex.models import ContentItem, IngestionReport


_MIN_DT = datetime.min.replace(tzinfo=timezone.utc)


class TwitterParser(SourceParser):
    platform = "twitter"

    def detect(self, root: Path) -> bool:
        data = root / "data"
        if not data.is_dir():
            return False

        try:
            load_js_array(data / "account.js")[0]["account"]["username"]
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            return False

        return (
            (data / "tweet.js").exists()
            or (data / "tweets.js").exists()
            or any(data.glob("tweets-part*.js"))
        )

    def parse(self, root: Path) -> Iterator[ContentItem]:
        report = IngestionReport(platform=self.platform, root=str(root))
        self.report = report
        t0 = time.perf_counter()

        data = root / "data"
        acct = load_js_array(data / "account.js")
        account = acct[0]["account"]
        own_id = account.get("accountId")
        own_handle = account.get("username")

        items: list[ContentItem] = []
        self._add_bio_item(data, own_handle, report, items)

        tweet_files = self._tweet_files(data)
        report.files_seen = len(tweet_files)

        authored: dict[str, dict] = {}
        seen: set[str] = set()
        for path in tweet_files:
            for row in iter_js_array(path):
                tweet = row.get("tweet") if isinstance(row, dict) else None
                if not isinstance(tweet, dict) or not tweet.get("id_str") or "full_text" not in tweet:
                    report.items_skipped_malformed += 1
                    continue

                tweet_id = tweet["id_str"]
                if tweet_id in seen:
                    continue
                seen.add(tweet_id)

                # TODO: validate vs real archive
                if tweet["full_text"].startswith("RT @"):
                    self._drop_noise(report, "retweet")
                    continue

                self_reply = self._is_self_reply(tweet, own_id, own_handle)
                if tweet.get("in_reply_to_status_id_str") is not None and not self_reply:
                    self._drop_noise(report, "reply_to_other")
                    continue

                text, quote_of_id, media_count = normalize_tweet(tweet)
                if not text:
                    report.items_skipped_empty += 1
                    continue

                authored[tweet_id] = {
                    "id": tweet_id,
                    "text": text,
                    "quote_of_id": quote_of_id,
                    "media_count": media_count,
                    "lang": tweet.get("lang"),
                    "created_at": parse_twitter_dt(tweet.get("created_at")),
                    "parent": tweet.get("in_reply_to_status_id_str") if self_reply else None,
                }

        for root_id, members in self._stitched_threads(authored):
            if len(members) > 1:
                item = self._thread_item(root_id, members, authored, own_handle)
                report.thread_members_folded += len(members) - 1
            else:
                item = self._post_item(root_id, authored[root_id], authored, own_handle)
            items.append(item)
            self._keep(report, item.content_type)

        report.duration_s = time.perf_counter() - t0
        return iter(items)

    def _add_bio_item(
        self,
        data: Path,
        own_handle: str | None,
        report: IngestionReport,
        items: list[ContentItem],
    ) -> None:
        profile_path = data / "profile.js"
        if not profile_path.exists():
            return

        try:
            profile = load_js_array(profile_path)
            description = profile[0]["profile"]["description"]
            bio = description.get("bio")
            text = " ".join(html.unescape(bio).split())
            if not text:
                return

            metadata = {}
            location = description.get("location")
            website = description.get("website")
            if location:
                metadata["location"] = location
            if website:
                metadata["website"] = website

            items.append(ContentItem(
                source_platform=self.platform,
                external_id="profile",
                content_type="bio",
                text=text,
                author_handle=own_handle,
                created_at=None,
                url=f"https://twitter.com/{own_handle}" if own_handle else None,
                metadata=metadata,
            ))
            self._keep(report, "bio")
        except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError):
            return

    def _tweet_files(self, data: Path) -> list[Path]:
        files = []
        for name in ("tweet.js", "tweets.js"):
            path = data / name
            if path.exists():
                files.append(path)
        files.extend(sorted(data.glob("tweets-part*.js")))
        return files

    def _is_self_reply(
        self,
        tweet: dict,
        own_id: str | None,
        own_handle: str | None,
    ) -> bool:
        if tweet.get("in_reply_to_status_id_str") is None:
            return False
        if own_id is not None:
            return tweet.get("in_reply_to_user_id_str") == own_id
        return own_handle is not None and tweet.get("in_reply_to_screen_name") == own_handle

    def _stitched_threads(self, authored: dict[str, dict]) -> list[tuple[str, list[str]]]:
        children: dict[str, list[str]] = {}
        for tweet_id, record in authored.items():
            parent = record["parent"]
            if parent is not None and parent in authored:
                children.setdefault(parent, []).append(tweet_id)

        roots = [
            tweet_id for tweet_id, record in authored.items()
            if record["parent"] is None or record["parent"] not in authored
        ]
        roots.sort(key=lambda tweet_id: self._sort_key(authored[tweet_id]))

        stitched = []
        for root_id in roots:
            member_ids = {root_id}
            stack = [root_id]
            while stack:
                current = stack.pop()
                for child_id in children.get(current, []):
                    if child_id not in member_ids:
                        member_ids.add(child_id)
                        stack.append(child_id)

            members = sorted(member_ids, key=lambda tweet_id: self._sort_key(authored[tweet_id]))
            stitched.append((root_id, members))
        return stitched

    def _thread_item(
        self,
        root_id: str,
        members: list[str],
        authored: dict[str, dict],
        own_handle: str | None,
    ) -> ContentItem:
        root = authored[root_id]
        return ContentItem(
            source_platform=self.platform,
            external_id=root_id,
            content_type="thread",
            text="\n\n".join(authored[member_id]["text"] for member_id in members),
            author_handle=own_handle,
            created_at=root["created_at"],
            url=f"https://twitter.com/{own_handle}/status/{root_id}" if own_handle else None,
            metadata={
                "lang": root["lang"],
                "thread_len": len(members),
                "member_ids": members,
                "media_count": sum(authored[member_id]["media_count"] for member_id in members),
            },
        )

    def _post_item(
        self,
        tweet_id: str,
        record: dict,
        authored: dict[str, dict],
        own_handle: str | None,
    ) -> ContentItem:
        metadata = {
            "lang": record["lang"],
            "is_reply": record["parent"] is not None and record["parent"] not in authored,
            "media_count": record["media_count"],
        }
        if record["quote_of_id"] is not None:
            metadata["quote_of_id"] = record["quote_of_id"]

        return ContentItem(
            source_platform=self.platform,
            external_id=tweet_id,
            content_type="post",
            text=record["text"],
            author_handle=own_handle,
            created_at=record["created_at"],
            url=f"https://twitter.com/{own_handle}/status/{tweet_id}" if own_handle else None,
            metadata=metadata,
        )

    def _sort_key(self, record: dict) -> tuple[datetime, str]:
        return (record["created_at"] or _MIN_DT, record["id"])

    def _keep(self, report: IngestionReport, content_type: str) -> None:
        report.items_kept += 1
        report.by_content_type[content_type] = report.by_content_type.get(content_type, 0) + 1

    def _drop_noise(self, report: IngestionReport, reason: str) -> None:
        report.items_dropped_noise += 1
        report.dropped_reasons[reason] = report.dropped_reasons.get(reason, 0) + 1
