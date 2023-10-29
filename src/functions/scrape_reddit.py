import logging
from typing import Tuple

import mlrun
import pandas as pd
import praw
from pydantic import BaseModel
from tqdm.auto import tqdm

logger = logging.getLogger("mlrun")

REMOVED = "[removed]"


class Comment(BaseModel):
    submission_id: str
    submission_title: str
    comment_id: str
    comment_body: str
    parent_comment_id: str


@mlrun.handler(outputs=["raw_comments", "formatted_replies"])
def scrape_subreddit_comments(
    subreddit: str, num_posts: int, filter_nsfw: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    reddit = praw.Reddit(
        client_id=mlrun.get_secret_or_env("REDDIT_CLIENT_ID"),
        client_secret=mlrun.get_secret_or_env("REDDIT_CLIENT_SECRET"),
        user_agent=mlrun.get_secret_or_env("REDDIT_USER_AGENT"),
        ratelimit_seconds=15,
    )

    comments = []
    for i, submission in enumerate(
        tqdm(reddit.subreddit(subreddit).hot(limit=num_posts))
    ):
        logger.info(f"Scraping submission {i}: {submission.title}")
        if filter_nsfw and submission.over_18:
            continue

        submission.comments.replace_more(limit=32, threshold=1)
        for comment in submission.comments.list():
            if comment.body == REMOVED:
                continue

            # Comment parent ids in form t3_175n2jt where 175n2jt is parent id
            _, parent_comment_id = comment.parent_id.split("_")

            comments.append(
                Comment(
                    submission_id=submission.id,
                    submission_title=submission.title,
                    comment_id=comment.id,
                    comment_body=comment.body,
                    parent_comment_id=parent_comment_id,
                )
            )

    # Dataframe of raw comments and submission info
    raw_comments = pd.DataFrame([c.dict() for c in comments])

    # Self join to get only entries with replies
    formatted_replies = pd.merge(
        raw_comments, raw_comments, left_on="comment_id", right_on="parent_comment_id"
    )[["comment_body_x", "comment_body_y"]]
    formatted_replies = formatted_replies.rename(
        {"comment_body_x": "comment", "comment_body_y": "reply"}, axis=1
    )

    # Formatted comment and reply pairs in following format
    # ### Comment: ... ### REPLY: ... ### END
    formatted_replies["formatted"] = (
        "### Comment: "
        + formatted_replies["comment"]
        + "### REPLY: "
        + formatted_replies["reply"]
        + " ### END"
    )

    return raw_comments, formatted_replies
