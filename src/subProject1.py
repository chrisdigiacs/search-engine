# Import necessary libraries
import nltk
import time
import json
import os
from itertools import chain
from bs4 import BeautifulSoup  # Used for parsing and navigating HTML/XML documents

def tokenizer(text):
    """
    Tokenizes a given text into words and abbreviations. It uses a regular expression
    to identify word patterns, including those with hyphens, apostrophes, and acronyms.

    Args:
        text (str): The text to be tokenized.

    Returns:
        list: A list of tokens (words and abbreviations) extracted from the text.
    """
     
    # Regular expression pattern to identify tokens
    pattern = r'\b(?:[A-Z]{1,2}\.)*[A-Z]{1,2}\.?\b|\b\w+(?:[-\']\w+)*\b'
    # Tokenize and return the list of tokens
    return nltk.regexp_tokenize(text, pattern)

def getTokens():
    """
    Retrieves and tokenizes documents from a specified corpus. Each .sgm file in the
    corpus is parsed, and documents within are tokenized. Tokens are associated with their
    document ID.

    Returns:
        list: A list where each element represents the tokens of a document, indexed by document ID.
    """

    tokensList = []
    # Inform about the start of token retrieval process
    print(f'Retrieving tokens ...')
    # Loop through each file in the corpus
    for file_number in range(22):
        with open(f'./Corpus/reut2-{str(file_number).zfill(3)}.sgm', 'r', encoding='windows-1252') as f:
            # Parse the file content using BeautifulSoup
            soup = BeautifulSoup(f, 'html.parser')
            documents = soup.find_all('text')
            
            for document in documents:
                doc_type = document.get('type')
                text_parts = []

                # Different handling for 'BRIEF' type documents
                if doc_type == 'BRIEF':
                    title = document.title.get_text() if document.title else ""
                    text_parts.append(title)
                else:
                    title = document.title.get_text() if document.title else ""
                    body = document.body.get_text() if document.body else ""
                    text_parts.append(title)
                    text_parts.append(body)
                
                # Combine title and body for tokenization
                full_text = ' '.join(text_parts)
                tokens = tokenizer(full_text)
                tokensList.append(tokens)

    # Indicate completion of token retrieval
    print(f'Tokens retrieved!')
    return tokensList

def spimi(tokensList, testCount=None):
    """
    Builds an inverted index using the Single-Pass In-Memory Indexing (SPIMI) algorithm. 
    It processes tokenized documents, generating a dictionary where each term is associated 
    with a list of tuples containing document IDs and term frequency.

    Args:
        tokensList (list): A list of lists, each containing tokens from a document.
        testCount (int, optional): A threshold for the number of term-document pairs to process.

    Returns:
        tuple: A tuple containing the inverted index and the time taken to build it.
    """
    index = {}
    docID = 1
    pairCount = 0
    # Print collection size and start building index
    print(f'Collection size: {len(tokensList)}')
    print(f'Building SPIMI index ...')
    startTime = time.time()
    for tokens in tokensList:
        for token in tokens:
            pairCount += 1
            # Update or add token in the index
            if token in index:
                postingsList = index[token]
                if postingsList[-1][0] == docID:
                    postingsList[-1] = (docID, postingsList[-1][1] + 1)
                else:
                    postingsList.append((docID, 1))
            else:
                index[token] = [(docID, 1)]

            if testCount and pairCount >= testCount:
                break
        docID += 1
        if testCount and pairCount >= testCount:
            break
    # Sort index by keys (terms) before saving
    index = dict(sorted(index.items(), key=lambda item: item[0]))
    endTime = time.time()

    print(f'SPIMI index built!')
    save2json(index, 'spimi.json')
    return index, endTime - startTime

def save2json(data, filename):
    """
    Saves a given data object to a JSON file.

    Args:
        data: The data to be saved.
        filename (str): The name of the file to save the data in.
    """

    # Create directory if it doesn't exist
    if not os.path.exists('indexes'):
        os.makedirs('indexes')
    
    # Save data to a JSON file
    with open(f'./indexes/{filename}', 'w') as f:
        json.dump(data, f, indent=4)

def naive(tokensList, testCount = None):
    """
    Builds an inverted index using a naive approach. It processes term-document pairs,
    sorts them, and then groups them to create postings lists.

    Args:
        tokensList (list): A list of tokenized documents.
        testCount (int, optional): A threshold for the number of term-document pairs to process.

    Returns:
        tuple: A tuple containing the inverted index and the time taken to build it.
    """

    F = []  # List to store term-docID pairs
    docID = 1
    print(f'Building Naive index ...')

    startTime = time.time()
    # Create term-docID pairs
    for tokens in tokensList:
        for token in tokens:
            F.append((token, docID))
            if testCount and len(F) == testCount:
                break
        docID += 1
        if testCount and len(F) == testCount:
            break

    # Sort and remove exact duplicates
    F.sort()
    F = list(dict.fromkeys(F))

    # Create postings lists
    index = {}
    for term, docID in F:
        if term in index:
            if index[term][-1] != docID:
                index[term].append(docID)
        else:
            index[term] = [docID]

    endTime = time.time()
    print(f'Naive index built!')
    save2json(index, 'naive.json')
    return index, endTime - startTime

def getStats(naiveIndex, naiveTime, spimiIndex, spimiTime, tokensTime):
    """
    Prints statistics comparing the Naive and SPIMI indexing methods. It displays the time taken
    to create token streams, build indexes, and the sizes of the created indexes.

    Args:
        naiveIndex (dict): The inverted index created using the naive approach.
        naiveTime (float): The time taken to build the naive index.
        spimiIndex (dict): The inverted index created using the SPIMI approach.
        spimiTime (float): The time taken to build the SPIMI index.
        tokensTime (float): The time taken to create token streams.
    """
    print('\n========== STATISTICS ==========\n')
    # Token streams creation time
    print(f'Token streams creation time (sec): {tokensTime}')
    # SPIMI Index Construction Statistics
    print('\n---------------------\n| SPIMI Construction |\n---------------------')
    print(f'\nTime taken for SPIMI index to process 10 000 terms (ms): {round(spimiTime * 1000, 3)}')
    print(f'Size of SPIMI index: {len(spimiIndex)} terms')
    # Naive Index Construction Statistics
    print('\n---------------------\n| Naive Construction |\n---------------------')
    print(f'\nTime taken for Naive index to process 10 000 terms (ms): {round(naiveTime * 1000, 3)}')
    print(f'Size of Naive index: {len(naiveIndex)} terms')
    # Differences between SPIMI and Naive Methods
    print('\n---------------\n| Differences |\n---------------')
    time_diff = naiveTime - spimiTime
    time_diff_percent = (time_diff / naiveTime) * 100
    print(f'\nTime difference (ms): {round(time_diff * 1000, 3)}')
    print(f'Time difference (%): {round(time_diff_percent, 3)}%')

def main():
    """
    Main function to execute the token retrieval, index building, and statistics generation process.
    It retrieves tokens, builds indexes using both naive and SPIMI approaches, and then prints statistics.
    """

    # Measure time taken to create token streams
    start = time.time()
    tokenStream = getTokens()
    end = time.time()
    tokensTime = round(end - start, 3)

    # Build SPIMI and Naive indexes
    spimiIndex, spimiTime = spimi(tokenStream, 10000)
    naiveIndex, naiveTime = naive(tokenStream, 10000)

    # Generate and print statistics
    getStats(naiveIndex, naiveTime, spimiIndex, spimiTime, tokensTime)

# Calling the main function to execute the program
if __name__ == "__main__":
    main()