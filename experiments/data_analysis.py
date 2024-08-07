import os
import matplotlib.pyplot as plt
from collections import Counter
from torchtext.data.utils import get_tokenizer
import nltk
from nltk.corpus import stopwords
import re

# Directory paths
train_dir = 'data/train' 
test_dir = 'data/test'  

# Tokenizer for bubble graph
tokenizer = get_tokenizer('basic_english')

# Irrelevant words to remove from the hbar chart
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))



# Extract rating from file name
def extract_rating(file_name):
    return int(file_name.split('_')[-1].split('.')[0])

# Remove punctuation and non-alphanumeric characters
def clean_text(text):
    # Remove <br> tags
    text = re.sub(r'<br\s*/?>', ' ', text)
    # Remove punctuation and non-alphanumeric characters
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

# Traverse directories and collect ratings
def collect_ratings_from_directory(directory):
    ratings = []
    for label in ['pos', 'neg']:
        label_dir = os.path.join(directory, label)
        for file_name in os.listdir(label_dir):
            if file_name.endswith('.txt'):
                rating = extract_rating(file_name)
                ratings.append(rating)
    return ratings

# Traverse directories and collect ratings and text length of each review
def collect_rating_and_text_lengths(data_dir):
    rating_by_len = []
    for label in ['pos', 'neg']:
        label_dir = os.path.join(data_dir, label)
        for file_name in os.listdir(label_dir):
            if file_name.endswith('.txt'):
                rating = extract_rating(file_name)
                file_path = os.path.join(label_dir, file_name)
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    cleaned_text = clean_text(text)
                    text_length = len(cleaned_text)
                    rating_by_len.append((rating, text_length))
    return rating_by_len

def plot_histogram_num_reviews_to_rating(ratings_list):

    # Count occurrences of each rating
    rating_counts = {}
    for rating in ratings_list:
        if rating in rating_counts:
            rating_counts[rating] += 1
        else:
            rating_counts[rating] = 1

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(rating_counts.keys(), rating_counts.values(), color='skyblue')
    plt.xlabel('Rating')
    plt.ylabel('Number of Reviews')
    plt.title('Histogram of Number of Reviews per Rating')
    plt.xticks(list(rating_counts.keys()))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_scatter_text_len_to_rating(data):
  
    # Separate ratings and text lengths for plotting
    ratings = [item[0] for item in data]
    text_lengths = [item[1] for item in data]


    # Intervals and their ranges
    intervals = [
    (0, 2000),
    (2000, 4000),
    (4000, 6000),
    (6000, 8000),
    (8000, 10000),
    (10000, 12000),
    (12000, 14000)  
    ]

    # Initialize counters for each interval
    interval_counts = [0] * len(intervals)

    # Count text lengths in each interval
    for length in text_lengths:
        for i, (start, end) in enumerate(intervals):
            if start <= length < end:
                interval_counts[i] += 1
                break  # Exit the inner loop once we've found the correct interval

    # Prepare data for scatter plot
    interval_names = [f"{start}-{end}" for start, end in intervals]
    counts = interval_counts

    # Plotting the scatter plot
    plt.figure(figsize=(14, 8))

    # Subplot for scatter plot of ratings vs. text lengths
    plt.subplot(1, 2, 1)
    plt.scatter(ratings, text_lengths, alpha=0.5, edgecolors='w', linewidth=0.5)
    plt.xlabel('Rating')
    plt.ylabel('Text Length')
    plt.title('Scatter Plot of Review Ratings vs. Text Lengths')
    plt.grid(True)

    # Subplot for scatter plot of text length intervals vs. counts
    plt.subplot(1, 2, 2)
    plt.scatter(interval_names, counts, s=100, c='blue', alpha=0.75)
    for i, count in enumerate(counts):
        plt.text(interval_names[i], count + 0.1, str(count), ha='center', va='bottom')
    plt.xlabel('Text Length Intervals')
    plt.ylabel('Number of Text Lengths')
    plt.title('Distribution of Text Lengths')
    plt.grid(True)

    plt.tight_layout()
    plt.show()



def traverse_and_tokenize_reviews(data_dir):
    word_counter_pos = Counter()
    word_counter_neg = Counter()
    for label in ['pos', 'neg']:
        label_dir = os.path.join(data_dir, label)
        for file_name in os.listdir(label_dir):
            if file_name.endswith('.txt'):
                file_path = os.path.join(label_dir, file_name)
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    cleaned_text = clean_text(text)
                    words = tokenizer(cleaned_text)
                    filtered_words = [word for word in words if word.lower() not in stop_words]
                    if label == 'pos':
                        word_counter_pos.update(filtered_words)
                    else:
                        word_counter_neg.update(filtered_words)

    return word_counter_pos, word_counter_neg

def plot_Hbar_of_word_count(all_word_counts, num_top_words):
    
    common_words = all_word_counts.most_common(num_top_words)
    
    # Extract data for plotting
    words, counts = zip(*common_words)
    words, counts = list(words), list(counts)

    # Sort words and counts in descending order
    words.reverse()
    counts.reverse()
        
    # Plotting the horizontal bar chart
    plt.figure(figsize=(15, 10))
    bars = plt.barh(words, counts, color='grey')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.title('Horizontal Bar Chart of Word Frequencies in Reviews (Excluding Stop Words and Punctuation)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Adding the frequency labels at the end of each bar
    for bar, count in zip(bars, counts):
        plt.text(count, bar.get_y() + bar.get_height() / 2, f'{count}', va='center')

    plt.show()



if __name__ == "__main__":


    # Collect ratings from train and test directories
    train_ratings = collect_ratings_from_directory(train_dir)
    test_ratings = collect_ratings_from_directory(test_dir)

    # Combine train and test ratings
    all_ratings = train_ratings + test_ratings

    plot_histogram_num_reviews_to_rating(all_ratings)



    # Collect data from train and test directories
    train_data = collect_rating_and_text_lengths(train_dir)
    test_data = collect_rating_and_text_lengths(test_dir)

    # Combine train and test data
    all_data = train_data + test_data

    plot_scatter_text_len_to_rating(all_data)



    # Collect word counts from train and test directories
    train_word_counts_pos, train_word_counts_neg = traverse_and_tokenize_reviews(train_dir)
    test_word_counts_pos, test_word_counts_neg = traverse_and_tokenize_reviews(test_dir)

    # Combine word counts from train and test directories
    all_word_counts_pos = train_word_counts_pos + test_word_counts_pos
    all_word_counts_neg = train_word_counts_neg + test_word_counts_neg


    plot_Hbar_of_word_count(all_word_counts_pos, 50)
    plot_Hbar_of_word_count(all_word_counts_neg, 50)


