import praw
import pandas as pd
from tqdm import tqdm

reddit = praw.Reddit(
    client_id="WsH2mzukYSiFLtYATnkq5A",
    client_secret="HIR6kQSv7hbODotpnhsXN6KVqxA7iQ",
    user_agent="temp",
)


def scrape_posts_and_comments(subreddit_name, total_posts, posts_per_iteration):
    """

    :param subreddit_name:
    :param total_posts:
    :param posts_per_iteration:
    :return:
    """
    subreddit = reddit.subreddit(subreddit_name)
    posts_data = []
    scraped_posts = 0
    after = None

    try:
        with tqdm(total=total_posts, desc=f"Scraping for {subreddit_name}") as pbar:
            while scraped_posts < total_posts:
                submissions = subreddit.new(limit=posts_per_iteration, params={'after': after})
                submission_list = list(submissions)
                if not submission_list:
                    break

                for submission in submission_list:
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
                    if scraped_posts >= total_posts:
                        break

                if len(submission_list) > 0:
                    after = submission_list[-1].fullname
                else:
                    break

        return pd.DataFrame(posts_data)

    except Exception as err:
        print(err)
        pass


def number_of_subreddit(subreddit_name: str):
    """

    :param subreddit_name:
    :return:
    """
    print("The number of submissions for " + subreddit_name + "measuring...")
    subreddit = reddit.subreddit(subreddit_name)

    submission_count = 0

    for _ in subreddit.new(limit=None):
        submission_count += 1

    print(f"Total number of submissions for {subreddit_name} in r/{subreddit_name}: {submission_count}")

    return submission_count
