import os, sys
import re

# hardcoded paths - change if necessary
root = 'data'

# this one you need to download from the dataset
full_dataset = 'movie_lines.txt'

output_destination = 'selected_conversations.txt'
movie_selection = 'selected_movies.txt'

# separator used in the original dataset
separator = ' +++$+++ '

# movie ID file
MOVIE_ID = 0

# full conversation dataset file
MOVIE_ID_FULL = 2
# reverse indexing
CHARACTER_NAME = -2
CHARACTER_LINE = -1

# keep just these characters for simplicity (and utf8 breaking)
repl = r'[^A-Za-z0-9()\,!\?\'\`\. ]'


# regex replace
def filter(string):
    return re.sub(repl, '', string)


# from a movie ID string (e.g. M134), output the number (134)
def number_from_id(id):
    return int(id[1:])


# read just movie ID's, rest is for readability
def read_selected(path_to_selected_movies):
    selected_movies = set()

    with open(path_to_selected_movies, 'r') as infile:
        for line in infile:
            parts = line.strip().split(separator)
            selected_movies.add(parts[MOVIE_ID].strip())
    return selected_movies


# select and write to output file
def select_and_write(path_to_full_dataset, path_to_output, selected_movies):
    movies = {}

    with open(path_to_full_dataset, 'r', encoding="ISO-8859-1") as infile:

        for line in infile:

            parts = line.strip().split(separator)

            if parts[MOVIE_ID_FULL].strip() not in selected_movies:
                continue

            # take data and transform to tuple
            ID = parts[MOVIE_ID_FULL]
            char_name = parts[CHARACTER_NAME]
            char_line = parts[CHARACTER_LINE]

            tup = (number_from_id(ID), char_name, char_line)

            # add to map
            if ID not in movies:
                movies[ID] = []
            movies[ID].append(tup)

    with open(path_to_output, 'w') as out:
        for movie in movies:
            # sort by line number
            dialogue = sorted(movies[movie], key=lambda t: t[0])
            for n, name, text in dialogue:
                out.write(filter(name) + ':\n' + filter(text) + '\n\n')


def main():
    # uses hardcoded paths
    selection = os.path.join(root, movie_selection)
    selected_movies = read_selected(selection)

    dataset = os.path.join(root, full_dataset)
    output = os.path.join(root, output_destination)
    select_and_write(dataset, output, selected_movies)


if __name__ == '__main__':
    main()
