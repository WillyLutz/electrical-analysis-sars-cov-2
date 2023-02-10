import os

DISK = "/media/wlutz/TOSHIBA EXT/Electrical activity analysis/SARS-COV-2/Organoids/Willy 3"

DATA = os.path.join(DISK, "DATA")
RESULTS = os.path.join(DISK, "RESULTS/")
MODELS = os.path.join(DISK, "MODELS/")
DATASETS = os.path.join(DISK, "DATASETS/")
FIGURES_PAPER = os.path.join(RESULTS, "Figures README Paper")
STACHEL = os.path.join(DATA, "+ Stachel/")
NOSTACHEL = os.path.join(DATA, "- Stachel/")

CHANNELS = ["TimeStamp [µs]", "47 (ID=0) [pV]", "48 (ID=1) [pV]", "46 (ID=2) [pV]", "45 (ID=3) [pV]", "38 (ID=4) [pV]",
            "37 (ID=5) [pV]", "28 (ID=6) [pV]", "36 (ID=7) [pV]", "27 (ID=8) [pV]", "17 (ID=9) [pV]", "26 (ID=10) [pV]",
            "16 (ID=11) [pV]", "35 (ID=12) [pV]", "25 (ID=13) [pV]", "15 (ID=14) [pV]", "14 (ID=15) [pV]",
            "24 (ID=16) [pV]", "34 (ID=17) [pV]", "13 (ID=18) [pV]", "23 (ID=19) [pV]", "12 (ID=20) [pV]",
            "22 (ID=21) [pV]", "33 (ID=22) [pV]", "21 (ID=23) [pV]", "32 (ID=24) [pV]", "31 (ID=25) [pV]",
            "44 (ID=26) [pV]", "43 (ID=27) [pV]", "41 (ID=28) [pV]", "42 (ID=29) [pV]", "52 (ID=30) [pV]",
            "51 (ID=31) [pV]", "53 (ID=32) [pV]", "54 (ID=33) [pV]", "61 (ID=34) [pV]", "62 (ID=35) [pV]",
            "71 (ID=36) [pV]", "63 (ID=37) [pV]", "72 (ID=38) [pV]", "82 (ID=39) [pV]", "73 (ID=40) [pV]",
            "83 (ID=41) [pV]", "64 (ID=42) [pV]", "74 (ID=43) [pV]", "84 (ID=44) [pV]", "85 (ID=45) [pV]",
            "75 (ID=46) [pV]", "65 (ID=47) [pV]", "86 (ID=48) [pV]", "76 (ID=49) [pV]", "87 (ID=50) [pV]",
            "77 (ID=51) [pV]", "66 (ID=52) [pV]", "78 (ID=53) [pV]", "67 (ID=54) [pV]", "68 (ID=55) [pV]",
            "55 (ID=56) [pV]", "56 (ID=57) [pV]", "58 (ID=58) [pV]", "57 (ID=59) [pV]"]

CHANNELS_FREQ = ["Frequency [Hz]", "47 (ID=0) [pV]", "48 (ID=1) [pV]", "46 (ID=2) [pV]", "45 (ID=3) [pV]",
                 "38 (ID=4) [pV]", "37 (ID=5) [pV]", "28 (ID=6) [pV]", "36 (ID=7) [pV]", "27 (ID=8) [pV]",
                 "17 (ID=9) [pV]", "26 (ID=10) [pV]",
                 "16 (ID=11) [pV]", "35 (ID=12) [pV]", "25 (ID=13) [pV]", "15 (ID=14) [pV]", "14 (ID=15) [pV]",
                 "24 (ID=16) [pV]", "34 (ID=17) [pV]", "13 (ID=18) [pV]", "23 (ID=19) [pV]", "12 (ID=20) [pV]",
                 "22 (ID=21) [pV]", "33 (ID=22) [pV]", "21 (ID=23) [pV]", "32 (ID=24) [pV]", "31 (ID=25) [pV]",
                 "44 (ID=26) [pV]", "43 (ID=27) [pV]", "41 (ID=28) [pV]", "42 (ID=29) [pV]", "52 (ID=30) [pV]",
                 "51 (ID=31) [pV]", "53 (ID=32) [pV]", "54 (ID=33) [pV]", "61 (ID=34) [pV]", "62 (ID=35) [pV]",
                 "71 (ID=36) [pV]", "63 (ID=37) [pV]", "72 (ID=38) [pV]", "82 (ID=39) [pV]", "73 (ID=40) [pV]",
                 "83 (ID=41) [pV]", "64 (ID=42) [pV]", "74 (ID=43) [pV]", "84 (ID=44) [pV]", "85 (ID=45) [pV]",
                 "75 (ID=46) [pV]", "65 (ID=47) [pV]", "86 (ID=48) [pV]", "76 (ID=49) [pV]", "87 (ID=50) [pV]",
                 "77 (ID=51) [pV]", "66 (ID=52) [pV]", "78 (ID=53) [pV]", "67 (ID=54) [pV]", "68 (ID=55) [pV]",
                 "55 (ID=56) [pV]", "56 (ID=57) [pV]", "58 (ID=58) [pV]", "57 (ID=59) [pV]"]
