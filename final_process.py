import re
import json
import sys
import time
import string
from mpi4py import MPI


# data structure for storing all words
sentimentWords = {}
# data structure for 'phrasal' terms
phrasal_words = {}
# grid Id mapping to tweet counts and tweet scores
gridBoundary = {}
MASTER_RANK = 0


def main():

    # n processes - calls this function n times
    startTime = time.time()
    if len(sys.argv) != 4:
        print("Missing command-line arguments")
        # command line syntax errors
        sys.exit(2)

    # file names
    sentiment_filename = sys.argv[1]
    grid_filename = sys.argv[2]
    twitter_filename = sys.argv[3]

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # construct coordinates and sentiment words data structures including phrasal words
    getSentimentWords(sentiment_filename)
    getCoordinates(grid_filename)

    if rank == 0:
        print(f"Number of cores for this task is {size}\n")
        # Master processor
        master_tweet_processor(comm, twitter_filename)
    else:
        # Slave processor
        slave_tweet_processor(comm, twitter_filename)

    print("\n--- %s seconds ---\n" % (time.time() - startTime))

    return


def update_counts(counts, total_counts):
    """Combine total counts with counts to updated dictionary"""
    for gridID in counts:
        # if grid exists already, update counts
        if gridID in total_counts:
            total_counts[gridID]['tweetCount'] += counts[gridID]['tweetCount']
            total_counts[gridID]['tweetScore'] += counts[gridID]['tweetScore']

        else:
            # check if gridID exists in counts - grid key doesn't exist in total_group_counts so we can
            # initialise it
            result = {'tweetCount': counts[gridID]['tweetCount'], 'tweetScore': counts[gridID]['tweetScore']}
            total_counts[gridID] = result
            total_counts[gridID] = result

    return total_counts


def get_tweets(comm):
    """Returns data to master processor when asked for it."""
    total_slave_counts = {}
    processes = comm.Get_size()

    # Now ask all processes except ourselves to return counts
    for i in range(processes-1):
        # request sent to slave_tweet_processor function
        comm.send('return_data', dest=i+1, tag=i+1)
    # receive counts back from worker nodes and update total_counts_slaves
    for i in range(processes-1):
        # Receive data
        total_slave_counts = update_counts(comm.recv(source=i+1, tag=MASTER_RANK), total_slave_counts)

    return total_slave_counts


def master_tweet_processor(comm, filename):
    """Receives data from slave processors and combines it with the data it has processed.
    Prints the result of this information to the prompt.
    Printed results are the total count of the tweet lengths measured by characters."""

    # Read out tweets
    rank = comm.Get_rank()
    size = comm.Get_size()

    total_slave_counts = {}

    # get counts for rank = 0 - this function is only called in 'if' statement of main
    # returns a dictionary of counts
    counts_master = getResults(rank, filename, size)

    if size > 1:
        # getting tweets from slaves - see 'else' statement above
        total_slave_counts = get_tweets(comm)

        # turn everything off once data has been marshalled
        for i in range(size-1):
            # receive data
            comm.send('exit', dest=(i+1), tag=(i+1))

    if total_slave_counts:
        counts_master = update_counts(counts_master, total_slave_counts)

    spacing = " " * 4

    print(f"Cell{spacing}#Total Tweets{spacing}#Overall Sentiment Score")
    for grid in sorted(counts_master.keys()):
        count = counts_master[grid]['tweetCount']
        score = counts_master[grid]['tweetScore']
        print(f"{grid:<4}{count:^15}{score:^25}")

    return


def slave_tweet_processor(comm, filename):
    """Each slave processor will process this function. Each processor will have a rank
    and we pass this rank and the size into the getResults function essentially getting that
    core to process the data that we have allocated to it. It then, when asked by the get_tweets
    function, will return the counts (associated with the rank) to the get_tweets function."""
    rank = comm.Get_rank()
    size = comm.Get_size()

    # returns a dictionary of counts
    counts = getResults(rank, filename, size)
    # We wait to return our counts now that we have them
    while True:
        # receive request sent from get_tweets function
        in_comm = comm.recv(source=MASTER_RANK, tag=rank)
        # check if command is a string with instruction: 'return_data' or 'exit'
        if isinstance(in_comm, str):
            # 'return_data' reflects the parameter data passed in
            if in_comm in "return_data":
                # send data back
                comm.send(counts, dest=MASTER_RANK, tag=MASTER_RANK)
            # terminates that processor/core's program
            elif in_comm in "exit":
                exit(0)

    return


def getSentimentWords(filename):
    """Read in AFINN.txt - store as dictionary {word: score} and return the dictionary containing
    words and corresponding score"""

    global sentimentWords
    global phrasal_words

    unique_words = set()

    count = 0

    with open(filename, "r") as file:
        for line in file:
            count += 1
            # pattern for matching words
            pattern1 = r".*(?=\s\-?\d)"
            # pattern for matching sentiment score
            pattern2 = r"[-]?\d"
            match1 = re.search(pattern1, line)
            wordMatch = match1.group(0)

            # match sentiment score
            match2 = re.search(pattern2, line)
            # get number from match
            numMatch = match2.group()
            numMatch = int(numMatch)
            sentimentWords[wordMatch] = numMatch

    # creates set of unique words and dictionary of phrasal words for use in calculateSentimentScore
    for key in sentimentWords:
        word = key.split()
        if len(word) > 1:
            phrasal_words[key] = sentimentWords[key]
            # get individual word of phrasal words
            for w in word:
                unique_words.add(w)

    return


def getCoordinates(filename):
    """Reads in melbGrid.json data and gets coordinates value for the grids.
    Returns a dictionary of grid values and their corresponding (x,y) corner
    coordinates."""

    global gridBoundary
    with open(filename, "r") as f:
        grid = json.load(f)
    features = grid["features"]

    for i, feature in enumerate(features):
        gridID = feature["properties"]["id"]
        properties = feature["properties"]

        # get extremity coordinates of big square
        x_min = properties["xmin"]
        x_max = properties["xmax"]
        y_min = properties["ymin"]
        y_max = properties["ymax"]

        # get boundary values for each grid
        gridBoundary[gridID] = (x_min, x_max, y_min, y_max)

    return gridBoundary


def tweet_to_json(tweet):
    """Cleans string of 'nuisance' characters to create properly
    formatted JSON string for reading"""

    tweet = re.sub("}},(\r|\n)+", "}}", tweet)
    tweet = re.sub('"source": "<a.*?>.*?</a>"', '"source":""', tweet)
    tweet = re.sub(r"}}]}$", "}}", tweet)
    tweet = json.loads(tweet)

    return tweet


def getResults(rank, filename, processes):
    """Process Twitter data for given rank and number of processes.
    Read data into memory for given worker node and process the data.
    Return a dictionary of results for those tweets processed by the given processor."""

    results = {}
    total_tweets = 0

    with open(filename, "r") as file:
        for i, tweet in enumerate(file):
            line_count = i
            # read in each line and execute the below for those lines which correspond to the rank of the
            # core being executed
            if line_count % processes == rank:
                try:
                    tweet = tweet_to_json(tweet)
                    total_tweets += 1
                    # division of tasks  -> process only those tweets corresponding to the rank of the processor where the
                    # remainder is the row number divided by the number of processors (typically 8)
                    # remainder will be 0-7 -> corresponds to the rank
                    coordinates = tweet["value"]["geometry"]["coordinates"]
                    tweetLocation = getGrid(coordinates)
                    # tweet not in one of the grids, exclude it
                    if tweetLocation == "not_found":
                        continue
                    else:
                        # get text from tweet
                        text = tweet["value"]["properties"]["text"]
                        # compute score for tweet
                        score = calculateSentimentScore(text)
                        # if gridID already exists in results
                        if tweetLocation in results:
                            # increment number of tweets in that grid
                            results[tweetLocation]["tweetCount"] += 1
                            # add score to existing total score for that grid
                            results[tweetLocation]["tweetScore"] += score

                        # gridID not in results, initialise key
                        else:
                            # set list to [tweetCount, tweetScore]
                            result = {"tweetCount": 1, "tweetScore": score}
                            results[tweetLocation] = result
                except ValueError as v:
                    print("\nMalformed JSON in tweet ", i)
                    print(v)
                    print(f"Line of text is: {tweet}\n")

    return results


def calculateSentimentScore(tweet):
    """Takes (tweet) text and AFINN sentiment words and computes
    the sentiment score for a given tweet.
    Returns the score for the given tweet.

    LOGIC:
    - check for 'phrasal words' first
    - then split the tweet up
    - regex can match those words in the tweet
    - look up words in sentimentWords except those words stored in 'found'
    - reset 'found' for each tweet
    """

    total = 0
    # store phrasal words found in tweet
    found = []
    for word in phrasal_words:
        pattern = r"(?:(?<=\s)|(?<=^)|(?<=[\"\']))(\b{}\b)(?=\s|$|[?!\"\'.,]*)".format(word)
        matches = re.findall(pattern, tweet, flags=re.IGNORECASE)
        if matches:
            found.append(word)
            multiplier = len(matches)
            score = sentimentWords[word] * multiplier
            total += score

    # total to this point reflect score from 'phrasal' words matching
    # now split the tweet up on whitespace and iterate over list, finding those words whose pattern
    # matches the desired pattern. Look these words up in sentimentWords to get their score.

    # tweet text - stored as list
    tweet_words = tweet.split()
    for word in tweet_words:
        pattern2 = f"[\"\']*[a-zA-Z-\']+[?!\"\'.,]*"
        pattern_sub1 = r"^[\"\']*"
        # pattern_sub2 = r"[?!\"\'.,]*"
        match = re.match(pattern2, word)
        if match is not None:
            matched_word = match.group(0)
            new_word = re.sub(pattern_sub1, "", matched_word)
            new_word = new_word.strip(string.punctuation)
            new_word = new_word.lower()
            if new_word in found:
                continue
            elif new_word in sentimentWords:
                score = sentimentWords[new_word]
                total += score
            # word is not in sentiment words, skip it
            else:
                continue

    return total


def getGrid(location):
    """Takes dictionary of id's and associated (x,y) values in a dictionary
    and coordinates of the tweet and returns the grid ID in which the
    tweet occurs if it indeed occurs within one of the locations. If tweet is outside
    of the specific value then it is ignored.

    Version 2: this consider the extremity points."""

    tweet_x_coord = location[0]
    tweet_y_coord = location[1]
    for gridID in gridBoundary:
        x_min, x_max, y_min, y_max = gridBoundary[gridID]
        # for left extremity A1
        if gridID == "A1":
            if tweet_x_coord == x_min and y_min < tweet_y_coord <= y_max:
                return gridID
            elif x_min < tweet_x_coord <= x_max and y_min < tweet_y_coord <= y_max:
                return gridID
            else:
                continue

        elif gridID == "B1":
            if tweet_x_coord == x_min and y_min < tweet_y_coord <= y_max:
                return gridID
            elif x_min < tweet_x_coord <= x_max and y_min < tweet_y_coord <= y_max:
                return gridID
            else:
                continue

        elif gridID == "C1":
            if tweet_x_coord == x_min and y_min < tweet_y_coord <= y_max:
                return gridID
            elif x_min < tweet_x_coord <= x_max and tweet_y_coord == y_min:
                return gridID
            elif x_min < tweet_x_coord <= x_max and y_min < tweet_y_coord <= y_max:
                return gridID
            else:
                continue

        elif gridID == "C2":
            if tweet_y_coord == y_min:
                return gridID
            elif x_min < tweet_x_coord <= x_max and y_min < tweet_y_coord <= y_max:
                return gridID
            else:
                continue

        elif gridID == "D3":
            if tweet_x_coord == x_min and y_min < tweet_y_coord <= y_max:
                return gridID
            elif x_min < tweet_x_coord <= x_max and y_min == tweet_y_coord:
                return gridID
            elif x_min < tweet_x_coord <= x_max and y_min < tweet_y_coord <= y_max:
                return gridID
            else:
                continue

        elif gridID == "D4":
            if tweet_y_coord == y_min:
                return gridID
            elif x_min < tweet_x_coord <= x_max and y_min < tweet_y_coord <= y_max:
                return gridID
            else:
                continue

        elif gridID == "D5":
            if tweet_y_coord == y_min:
                return gridID
            elif x_min < tweet_x_coord <= x_max and y_min < tweet_y_coord <= y_max:
                return gridID
            else:
                continue

        elif x_min < tweet_x_coord <= x_max and y_min < tweet_y_coord <= y_max:
            return gridID
    gridID = "not_found"

    return gridID



if __name__ == "__main__":
    main()