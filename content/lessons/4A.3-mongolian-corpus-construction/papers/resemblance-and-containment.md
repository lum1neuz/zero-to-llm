---
paper: "On the resemblance and containment of documents"
gist: Turn each document into a set of overlapping word sequences, and two documents' similarity becomes a set overlap that a few random permutations can estimate cheaply.
---

## The problem {#problem}
> A web search engine needs to know which pages are copies or near-copies of each other, across millions of pages, without comparing every pair in full.

Broder worked on this for the AltaVista search engine. Pages get mirrored, edited lightly or wrapped in different headers, so exact comparison misses most copies, and comparing full texts pairwise is far too slow.

## Shingles {#shingles}
> Represent a document by the set of all its contiguous sequences of w words, its w-shingles.

Word order matters only locally: two documents that share many w-word runs are about the same text, even if things were added or moved around. Turning a document into a set lets you compare documents with set operations. 4A.3 uses 3-word shingles.

## Resemblance and containment {#measures}
> Resemblance is the shared shingles over all distinct shingles (Jaccard); containment is the shared shingles over one document's own shingles.

Resemblance near 1 means the two documents are roughly the same. Containment of A in B near 1 means A appears almost entirely inside B, even if B is much longer. That's the case 4A.3's Khural story shows. Doc 3 contains all of doc 2, but their Jaccard is only 0.57, because Jaccard counts the extra sentence against them.

## Min-wise permutations {#minwise}
> Under a random permutation of all possible shingles, two sets have the same smallest element with probability exactly equal to their resemblance.

The element of the union that comes first under the permutation is equally likely to be any element of the union. Both sets have it as their minimum exactly when it's in the intersection, which happens with probability |A ∩ B| / |A ∪ B|. Keep the minimum under many independent permutations and the fraction of agreements estimates the resemblance. In practice seeded hash functions stand in for random permutations, as in 4A.3's `h(seed, s)`.

## Sketches {#sketches}
> A short sketch per document, a fixed number of minimum hash values, is enough to estimate resemblance without keeping the documents.

Sketches are small and of fixed size whatever the document's length. Comparing two documents means comparing two short lists. Grouping sketches cleverly, which later became locality-sensitive hashing, avoids comparing every pair at all.

## How the course uses it {#course}
> 4A.3's shingles, Jaccard and MinHash steps are this paper's ideas, written out in a few lines of Python.

Your `minhash` is a sketch of 64 minimums, one per seed, and `estimate` is the fraction of positions that agree. On the mirrored weather report it gives 0.83 against an exact Jaccard of 0.82.
