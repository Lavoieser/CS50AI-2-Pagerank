import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])

    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")

    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    len_corpus = len(corpus)
    nb_links = len(corpus[page])
    transition = dict()
    for link in corpus:
        probability = 1 / len_corpus * (1 - damping_factor)
        if link in corpus[page]:
            probability = probability + damping_factor * 1 / nb_links
        transition[link] = probability

    rand = random.choices(
        population=list(transition.keys()),
        weights=list(transition.values()),
        k=1
    )[0]

    return rand


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Generate a first page at random in the corpus
    next_page = random.choices(
        list(corpus.keys()),
        k=1
    )[0]

    sample = dict()
    for i in range(n):
        next_page = transition_model(corpus, next_page, DAMPING)
        if next_page in sample:
            sample[next_page] += 1 / n
        else:
            sample[next_page] = 1 / n

    sorted_sample = {k: sample[k] for k in sorted(sample.keys())}

    return sorted_sample


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    limit = 0.0001
    N = len(corpus)
    PR = dict()
    # Assign each page a starting PageRank value of 1 / N
    for page in corpus:
        PR[page] = 1 / N

    # A page that has no links at all should be interpreted as having one link for every page in the corpus (including itself).
    for page in corpus:
        if len(corpus[page]) == 0:
            corpus[page] = set(corpus.keys())

    # Iterate until convergence
    PR_next = dict(PR)
    incomplete = True
    while incomplete:
        PR = dict(PR_next)
        for page in corpus:
            PR_page_new = 0
            delta_max = 0
            for i in corpus:    # Iterate all pages to find those referring to the page
                if page in corpus[i]:
                    PR_page_new = PR_page_new + PR[i] / len(corpus[i])

            if abs(PR_page_new - PR[page]) > delta_max:
                delta_max = abs(PR_page_new - PR[page])

            PR_next[page] = PR_page_new

        if delta_max < limit:
            incomplete = False

    # Complete PR calculation
    for page in corpus:
        PR[page] = damping_factor * PR[page] + (1 - damping_factor) / N

    return PR


if __name__ == "__main__":
    main()
