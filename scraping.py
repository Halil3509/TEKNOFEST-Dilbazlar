import praw
import pandas as pd
from tqdm import tqdm
import time

reddit = praw.Reddit(
    client_id="WsH2mzukYSiFLtYATnkq5A",
    client_secret="HIR6kQSv7hbODotpnhsXN6KVqxA7iQ",
    user_agent="temp",
)


def scrape_posts_and_comments(subreddit_name):
    """
    Only scraping side of submissions and their comments and make them a tabular data.

    :param subreddit_name:
    :return:
    """
    subreddit = reddit.subreddit(subreddit_name)
    posts_data = []
    scraped_posts = 0

    try:

        submission_list = list(subreddit.new(limit=None))
        print("After New: ", len(submission_list))
        submission_list.extend(list(subreddit.hot(limit=None)))
        print("After Hot: ", len(submission_list))
        submission_list.extend(list(subreddit.top(limit=None)))
        print("After Top: ", len(submission_list))
        submission_list.extend(list(subreddit.rising(limit=None)))
        print("After Rising: ", len(submission_list))

        # Get unique values
        print(f"Total value of {subreddit_name}:", len(set(submission_list)))

        with tqdm(total=len(set(submission_list)), desc=f"Scraping for {subreddit_name}") as pbar:
            for submission in list(set(submission_list)):
                submission.comments.replace_more(limit=0)
                comments = submission.comments.list()

                for comment in comments:
                    posts_data.append({
                        'post_id': submission.id,
                        'post_title': submission.title,
                        'post_body': submission.selftext,
                        'post_score': submission.score,
                        'post_url': submission.url,
                        'post_created': submission.created_utc,
                        'comment_id': comment.id,
                        'comment_body': comment.body,
                        'comment_score': comment.score,
                        'comment_created': comment.created_utc,
                    })

                scraped_posts += 1
                pbar.update(1)

                if scraped_posts % 500 == 0:
                    print("Breath Dude")
                    time.sleep(120)

    except Exception as err:
        print(err)
        pass

    return pd.DataFrame(posts_data)
