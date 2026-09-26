"""Download the LinkedIn corpus of Ayoobi et al. (2023) from its official repository and verify it.

The corpus is not redistributed here: its repository carries no licence file and asks users to
cite the original paper. This script fetches the file from the source and checks that it is the
exact release used in our experiments.

Usage:  python scripts/download_data.py            (writes data/LinkedIn_Dataset.pcl)
"""
import hashlib, os, sys, urllib.request

URL = "https://raw.githubusercontent.com/navid-aub/LinkedIn-Dataset/main/LinkedIn_Dataset.pcl"
SHA256 = "6d67c8015fceee06226491a8b0683bfc01e220416fed99e7f2d91aca27ba2e1a"
EXPECTED = {0: 1800, 1: 600, 10: 600, 11: 600}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "LinkedIn_Dataset.pcl")

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if not os.path.exists(OUT):
        print("downloading", URL)
        urllib.request.urlretrieve(URL, OUT)
    digest = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
    if digest != SHA256:
        sys.exit("checksum mismatch: got %s, expected %s. The upstream file may have changed." % (digest, SHA256))
    print("checksum OK (%s)" % SHA256)
    # the file is a pandas pickle; unpickle it only because its checksum matches the known release
    import pandas as pd
    df = pd.read_pickle(OUT)
    counts = df["Label"].value_counts().sort_index().to_dict()
    if counts != EXPECTED:
        sys.exit("unexpected label counts: %s" % counts)
    print("rows %d, columns %d, label counts %s" % (df.shape[0], df.shape[1], counts))
    print("saved to", os.path.normpath(OUT))

if __name__ == "__main__":
    main()
