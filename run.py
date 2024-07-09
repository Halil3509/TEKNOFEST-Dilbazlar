from scraping import scrape_posts_and_comments, number_of_subreddit
from llm_filtering import filter_using_llm
from tqdm import tqdm

subreddit_names = ["generalizedanxiety"]  # "Agoraphobia" was already scraped

for subreddit_name in tqdm(subreddit_names, desc="Reddit Scraping"):
    number_of_submissions = number_of_subreddit(subreddit_name)

    # Scraping
    df = scrape_posts_and_comments(subreddit_name=subreddit_name, total_posts=number_of_submissions, posts_per_iteration=100)

    llm_filtered_df = filter_using_llm(df=df)

    df.to_csv(f"Data/{subreddit_name}_pure_en.csv", index=False)
    llm_filtered_df.to_csv(f"Data/{subreddit_name}_llm_filtered_en.csv", index=False)
    print(f"{subreddit_name} was saved.")
