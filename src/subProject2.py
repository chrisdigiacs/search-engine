import nltk
import time
import os
import math
from itertools import chain
from subProject1 import spimi, naive, tokenizer, getTokens

def queryTest(query, spimiIndex=None, naiveIndex=None, collection=None, operation='OR', queryTermRanking=False, bm25Ranking=False):
    """
    Processes a search query against provided SPIMI and/or Naive indexes. It supports single or multi-term
    queries with AND/OR operations and optional ranking (Query Term Ranking or BM25).

    Args:
        query (str): The search query.
        spimiIndex (dict): The SPIMI index (optional).
        naiveIndex (dict): The Naive index (optional).
        collection (list): The document collection (used for BM25 ranking).
        operation (str): The boolean operation to apply ('AND', 'OR').
        queryTermRanking (bool): Flag to use Query Term Ranking.
        bm25Ranking (bool): Flag to use BM25 ranking.

    The function tokenizes the query, checks the appropriate index for each term, and then applies the
    specified operation (AND/OR) to compute the final result set. It also handles ranking if specified.
    """
    # Tokenize the input query
    queryTerms = tokenizer(query)

    # Check if the query is empty after tokenization
    if len(queryTerms) == 0:
        print(f'Sorry, \'{query}\' is not a valid query')
        return

    # Process with Naive Index if available
    if naiveIndex:
        print("\n====== NAIVE INDEX ======\n")
        # Process single-term queries
        if len(queryTerms) == 1:
            result = naiveIndex.get(queryTerms[0])  # Get postings for the single query term.
        else:
            # Process multi-term queries
            postingsLists = []
            result = None
            for term in queryTerms:
                postingsList = naiveIndex.get(term)  # Get postings for each query term.
                postingsLists.append(postingsList)
            # Apply boolean operations
            if operation == 'AND':
                result = conjunction(postingsLists)
            elif operation == 'OR':
                result = disjunction(postingsLists, queryTermRanking)
        
        # Print the results from Naive Index
        if result:
            print(f"QUERY: \'{query}\'\nOPERATION: {operation}\nRANKING: {'Query Term Ranking' if queryTermRanking else 'None'} \nPOSTINGS LIST SIZE: {len(result)}\nPOSTINGS LIST: {str(result)}\n")
        else:
            print(f"QUERY: \'{query}\' - Your query returned no results with operation: {operation}.\n")

    # Process with SPIMI Index if available
    if spimiIndex:
        print("\n====== SPIMI INDEX ======\n")
        # Process single-term queries
        if len(queryTerms) == 1:
            result = spimiIndex.get(queryTerms[0])  # Get postings for the single query term.
        else:
            # Process multi-term queries
            postingsLists = []
            result = None
            for term in queryTerms:
                postingsList = spimiIndex.get(term)  # Get postings for each query term.
                postingsLists.append(postingsList)
            # Apply boolean operations
            if operation == 'AND':             
                normalized = convertPostingsLists(postingsLists)
                result = conjunction(normalized)
            elif operation == 'OR':
                normalized = convertPostingsLists(postingsLists)
                result = disjunction(normalized, queryTermRanking)
            # Apply BM25 ranking if enabled
            if bm25Ranking and result:
                k1, b = getK1B()
                result = bm25(queryTerms, result, spimiIndex, collection, k1, b)

        # Print the results from SPIMI Index
        if result:
            print(f"QUERY: \'{query}\'\nOPERATION: {operation}\nRANKING: {'Query Term Ranking' if queryTermRanking else 'BM25' if bm25Ranking else 'None'}\nPOSTINGS LIST SIZE: {len(result)}\nPOSTINGS LIST: {str(result)}\n")
        else:
            print(f"QUERY: \'{query}\' - Your query returned no results with operation: {operation}.\n")    

def getK1B():
    """
    Prompts the user to input values for the BM25 parameters k1 and b, ensuring they fall within valid ranges.
    The function loops until valid inputs are provided.

    Returns:
        tuple: A tuple (k1, b) with the BM25 parameters.
    """
    # Get k1 value
    while True:
        try:
            k1 = float(input('Enter your k1 value (k1 >= 0): '))
            # Check if k1 is in the valid range
            if k1 >= 0:
                break
            else:
                print(f'Sorry, {k1} is not a valid k1 value. It must be greater than or equal to 0.')
        except ValueError:
            # Handle invalid input
            print('Invalid input. Please enter a number.')

    # Get b value
    while True:
        try:
            b = float(input('Enter your b value (0 <= b <= 1): '))
            # Check if b is in the valid range
            if 0 <= b <= 1:
                break
            else:
                print(f'Sorry, {b} is not a valid b value. It must be between 0 and 1.')
        except ValueError:
            # Handle invalid input
            print('Invalid input. Please enter a number.')

    # Return the k1 and b values
    return k1, b

def bm25(queryTerms, result, index, collection, k1, b):
    """
    Applies the BM25 ranking formula to the result set of a query. It calculates scores for documents
    based on term frequency, document frequency, and document length.

    Args:
        queryTerms (list): List of tokenized query terms.
        result (list): List of document IDs obtained from the initial search.
        index (dict): The search index.
        collection (list): The document collection, to compute document lengths.
        k1 (float): BM25 parameter k1.
        b (float): BM25 parameter b.

    Returns:
        list: A sorted list of tuples (docID, score), ranked according to BM25.
    """
    N = len(collection)  # Total number of documents
    L_total = sum(len(doc) for doc in collection)  # Total length of all documents
    L_avg = L_total / N  # Average document length

    rankedResults = {}

    if result is None:
        return None

    # Calculate BM25 score for each term in each document
    for term in queryTerms:
        postingsList = index.get(term, [])
        df = len(postingsList)  # Document frequency
        idf = math.log(N / df) if df != 0 else 0  # Inverse document frequency

        for docID, tf in postingsList:
            if docID and docID in result:
                L_d = len(collection[int(docID)])  # Document length for the current docID
                # BM25 formula
                score = idf * ((tf * (k1 + 1)) / (k1 * ((1 - b) + b * (L_d / L_avg)) + tf))
                # Accumulate score for each document
                if docID not in rankedResults:
                    rankedResults[docID] = 0.0 
                rankedResults[docID] += score

    # Check if all scores are zero
    if all(score == 0.0 for score in rankedResults.values()):
        return None

    # Sort the results by score in descending order
    sortedRankedList = sorted(rankedResults.items(), key=lambda item: item[1], reverse=True)
    return sortedRankedList

def convertPostingsLists(postingsLists):
    """
    Convert a list of postings lists with (docID, tf) tuples to a list of lists of docIDs.
    If a postings list is None, it remains None.

    Args:
        postingsLists (list): A list of postings lists, where each postings list contains tuples of (docID, tf).

    Returns:
        list: A list of lists, where each inner list contains docIDs or is None.
    """
    # Convert each postings list to a list of docIDs, or keep as None if postings list is None
    return [[docID for docID, tf in postingsList] if postingsList is not None else None for postingsList in postingsLists]

def conjunction(postingsLists):
    """
    Performs an AND operation on a list of postings lists, returning the intersection of these lists.

    Args:
        postingsLists (list of lists): A list where each element is a postings list (list of docIDs).

    Returns:
        list: The intersection of the postings lists, representing documents that contain all terms.
    """
    # Return None if any of the postings lists is None
    if None in postingsLists:
        return None

    # Sort postings lists by length for efficient intersection
    sortedPostingsLists = sorted(postingsLists, key=len)
    # Start with the shortest postings list
    finalPostingsList = sortedPostingsLists[0]

    # Intersect with each subsequent postings list
    for postingsList in sortedPostingsLists[1:]:
        finalPostingsList = intersect(finalPostingsList, postingsList)
        # If intersection is empty, no further processing is needed
        if not finalPostingsList:
            break
    return finalPostingsList

def disjunction(postingsLists, queryTermRanking=False):
    """
    Performs an OR operation on a list of postings lists, returning the union of these lists. Can also
    apply query term ranking based on term frequency across the postings lists.

    Args:
        postingsLists (list of lists): A list where each element is a postings list (list of docIDs).
        queryTermRanking (bool): Flag to apply query term ranking.

    Returns:
        list: The union of the postings lists, with optional query term ranking applied.
    """
    # Filter out None postings lists (terms not found)
    validPostingsLists = [plist for plist in postingsLists if plist is not None]

    # If all postings lists are None, return None
    if not validPostingsLists:
        return None
    
    # Flatten the list of lists into a single list using itertools.chain
    unionPostings = list(chain(*validPostingsLists))
    
    # If query term ranking is not applied, remove duplicates and sort
    if not queryTermRanking:
        finalPostings = sorted(set(unionPostings))
    else:
        # Sort by document ID and apply query term ranking
        finalPostings = sorted(unionPostings)
        finalPostings = queryTermRank(finalPostings)

    return finalPostings        

def intersect(pList1, pList2):
    """
    Computes the intersection of two postings lists.

    Args:
        pList1 (list): The first postings list.
        pList2 (list): The second postings list.

    Returns:
        list: The intersection of pList1 and pList2.
    """
    result = []
    i = j = 0
    # Iterate through both lists to find common elements
    while i < len(pList1) and j < len(pList2):
        if pList1[i] == pList2[j]:
            result.append(pList1[i])
            i += 1
            j += 1
        elif pList1[i] < pList2[j]:
            i += 1
        else:
            j += 1
    return result

def queryTermRank(postingsList):
    """
    Applies query term ranking to a postings list. Ranks documents based on the number of query terms they contain.

    Args:
        postingsList (list): A postings list (list of docIDs).

    Returns:
        list: A list of tuples (docID, score) where score is the count of query terms in the document.
    """
    # Count the frequency of each document ID in the postings list
    docFreq = {}
    for docID in postingsList:
        docFreq[docID] = docFreq.get(docID, 0) + 1
    
    # Sort documents by frequency (score) in descending order
    rankedDocs = sorted(docFreq.items(), key=lambda item: item[1], reverse=True)
    return [(docID, score) for docID, score in rankedDocs]

def getOperation():
    """
    Prompts the user to choose a boolean operation ('AND' or 'OR') for the query processing.

    Returns:
        str: The chosen boolean operation ('AND' or 'OR').
    """
    valid = False
    while not valid:
        operation = input('Enter your operation (\'AND\', \'OR\') : ')
        # Check if the entered operation is valid
        if operation in ['AND', 'OR']:
            valid = True
        else:
            print(f'Sorry, {operation} is not a valid operation.')
    return operation

def getIndexes(spimiIndex, naiveIndex):
    """
    Prompts the user to choose which index(es) to query - SPIMI, Naive, or both.

    Args:
        spimiIndex (dict): The SPIMI index.
        naiveIndex (dict): The Naive index.

    Returns:
        tuple: A tuple of the selected indexes. None is used for unselected indexes.
    """
    valid = False
    while not valid:
        answer = input('Which index would you like to query? \'n\' = naive, \'s\' = spimi, \'b\' = both : ')
        # Validate the user input and return the appropriate indexes
        if answer in ['n', 's', 'b']:
            valid = True
        else:
            print(f'Sorry, {answer} is not a valid answer.')
    # Return the appropriate combination of indexes based on user choice
    if answer == 's':
        return spimiIndex, None
    elif answer == 'n':
        return None, naiveIndex
    elif answer == 'b':
        return spimiIndex, naiveIndex

def getRanking(operation, spimiIndex):
    """
    Prompts the user to choose a ranking method for the query results.

    Args:
        operation (str): The chosen boolean operation ('AND' or 'OR').
        spimiIndex (dict): The SPIMI index.

    Returns:
        tuple: A tuple (queryTermRank, bm25) indicating whether query term ranking or BM25 should be used.
    """
    queryTermRank = False
    bm25 = False
    valid = False
    while not valid:
        answer = input('Which ranking would you like? \'q\' = query term ranking, \'b\' = bm25, \'n\' = none : ')
        # Validate the user input and set the ranking method accordingly
        if answer in ['q', 'b', 'n']:
            if answer == 'q' and operation != 'OR':
                print('Sorry, query term ranking is only for multi-term OR queries.')
            elif answer == 'b' and not spimiIndex:
                print('Sorry, bm25 ranking is only available for the spimi index.')
            else:
                valid = True
                if answer == 'q':
                    queryTermRank = True
                elif answer == 'b':
                    bm25 = True
        else:
            print(f'Sorry, {answer} is not a valid answer.')
    
    return queryTermRank, bm25

def queryManager(spimiIndex, naiveIndex, collection):
    """
    Manages the querying process. It prompts the user to enter queries and handles the retrieval
    and display of results using the chosen index and ranking method.

    Args:
        spimiIndex (dict): The SPIMI index.
        naiveIndex (dict): The Naive index.
        collection (list): The document collection, used for BM25 ranking.

    This function continues to prompt for queries until the user chooses to quit.
    """
    print('\n====== WELCOME TO THE REUTERS21578 SEARCH ENGINE ======\n')
    print('\n** Enter \'q\' to quit **\n')
    print('** Queries do not require boolean operators **\n')
    while True:        
        query = input('Enter your query: ')
        if query == 'q':
            break
        numKeywords = len(tokenizer(query))
        operation = getOperation() if numKeywords > 1 else None
        spimi, naive = getIndexes(spimiIndex, naiveIndex)
        queryTermRanking, bm25Ranking = getRanking(operation, spimi)
        queryTest(query=query, spimiIndex=spimi, naiveIndex=naive, collection=collection, operation=operation, queryTermRanking=queryTermRanking, bm25Ranking=bm25Ranking)
    print('\n====== SEE YOU LATER :P ======\n')

def main():
    """
    The main function to initiate the search engine application. It retrieves token streams, builds indexes,
    and launches the query manager to handle user queries.

    This function is the starting point of the application.
    """
    tokenStreams = getTokens()
    spimiIndex, _ = spimi(tokenStreams)
    naiveIndex, _ = naive(tokenStreams)
    queryManager(spimiIndex, naiveIndex, tokenStreams)

if __name__ == "__main__":
    main()