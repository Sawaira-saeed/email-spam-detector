"""Dataset loader for the SMS Spam Collection dataset."""

from __future__ import annotations

import io
import os
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_FILE = DATA_DIR / "spam.csv"

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
GITHUB_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

SAMPLE_DATA = [
    ("ham", "Go until jurong point, crazy.. Available only in bugis n great world la e buffet.. Cine there got amore wat..."),
    ("ham", "Ok lar... Joking wif u oni..."),
    ("spam", "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"),
    ("ham", "U dun say so early hor... U c already then say..."),
    ("ham", "Nah I don't think he goes to usf, he lives around here though"),
    ("spam", "FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it later? Reply with yes if interested."),
    ("ham", "Even my brother is not like that"),
    ("ham", "As per your request 'Melle Melle (Oru Minnaminunginte Netttanttru) is the movie name. Please note that the correct movie name is Melle Melle (Oru Minnaminunginte Netttanttru) and not Melle Melle (Oru Minnaminunginte Netttanttru)."),
    ("spam", "WINNER!! You have been selected to receive a cash prize of £500. Reply with CLAIM to claim your prize."),
    ("ham", "I'm going to be home soon and i don't want to think of such things. You are my only love"),
    ("ham", "I have a date on sunday with will!!"),
    ("spam", "URGENT! You have won a 1 week FREE membership to our Man`s Unforgetable SMS to claim code XXXXX std txt rate to claim"),
    ("ham", "Oh k...i'm watching here:)"),
    ("ham", "Eh u remember which walkin we were talking about? Which one is that?"),
    ("spam", "Six months free mobile line rental on the latest Samsung or Sony Ericsson Bluetooth camera phone with 1.3 Megapixel camera, Java, colour screen, polyphonic tones, video player & MMS. 200 min cross ntwk calls & 100 txts. To claim call 08002986030"),
    ("ham", "That's gr8 :) I'm so happy for you. Congrats"),
    ("ham", "No I'm not. I'm still at home. My mom told me to stay home today."),
    ("spam", "You have won a Nokia 3650 video phone. To claim, call 09066363636. Claims must be made by 10th Sept. 1 claim per person. 150p/min after first 2 min. 18+ only."),
    ("ham", "I'll text you when I'm on the train."),
    ("ham", "What time do you want to meet?"),
    ("spam", "Congratulations! You've won a $1,000 Walmart gift card. Go to http://bit.ly/claim123 to claim now."),
    ("ham", "Hey, are we still on for dinner tonight?"),
    ("ham", "Thanks for the birthday wishes! Really appreciate it."),
    ("spam", "URGENT: Your mobile number has been selected for a $2000 prize. Call 09061701461 to claim."),
    ("ham", "Can you pick up some milk on your way home?"),
    ("ham", "I just got back from the gym. Feeling great!"),
    ("spam", "FREE RINGTONE text SOUNDS to 87077 to get a free polyphonic ringtone. T&C apply."),
    ("ham", "See you at the meeting tomorrow at 10am."),
    ("ham", "The weather is beautiful today. Perfect for a walk."),
    ("spam", "You are a winner! Claim your £500 prize by calling 09061701461 now."),
    ("ham", "Don't forget to bring the documents."),
    ("ham", "How was your day? Mine was pretty good."),
    ("spam", "Click here to claim your free iPhone: http://bit.ly/freeiphone"),
    ("ham", "I love you too sweetheart <3"),
    ("ham", "Mom called, she wants to talk to you."),
    ("spam", "LIMITED TIME OFFER! Buy one get one free on all items. Text YES to subscribe."),
    ("ham", "The package has been delivered to your doorstep."),
    ("ham", "Are you coming to the party this weekend?"),
    ("spam", "You have 1 new message from a secret admirer. Call 08718718721 to listen."),
    ("ham", "Good morning! Have a great day at work."),
    ("ham", "I finished the report and sent it to the boss."),
    ("spam", "Earn $500/day from home. No experience needed. Visit www.earnathome.com"),
    ("ham", "Let's grab lunch sometime next week."),
    ("ham", "The kids are doing well in school."),
    ("spam", "URGENT: We are trying to contact U. Last weekends draw shows U have won £800. Call 09061701461."),
    ("ham", "I'm at the grocery store, need anything?"),
    ("ham", "Thanks for the help earlier, really appreciate it."),
    ("spam", "Your account has been credited with 500 points. Redeem now at www.rewards.com"),
    ("ham", "Looking forward to the weekend!"),
    ("ham", "Did you watch the game last night? It was incredible!"),
]


def download_dataset() -> bool:

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    sources = [
        (GITHUB_URL, "tsv"),
        (UCI_URL, "zip"),
    ]

    for url, fmt in sources:
        try:
            logger.info("Attempting download from: %s", url)
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            if fmt == "tsv":
                df = pd.read_csv(
                    io.BytesIO(response.content),
                    sep="\t",
                    header=None,
                    names=["label", "message"],
                    encoding="utf-8",
                )
            elif fmt == "zip":
                z = zipfile.ZipFile(io.BytesIO(response.content))
                filename = z.namelist()[0]
                content = z.read(filename).decode("utf-8", errors="replace")
                lines = [line.strip().split("\t", 1) for line in content.strip().split("\n") if line.strip()]
                df = pd.DataFrame(lines, columns=["label", "message"])

            df = df[df["label"].isin(["ham", "spam"])].reset_index(drop=True)
            df.to_csv(DATA_FILE, index=False)
            logger.info("Dataset saved: %d rows", len(df))
            return True

        except Exception as exc:
            logger.warning("Download from %s failed: %s", url, exc)
            continue

    return False


def load_dataset() -> pd.DataFrame:

    if DATA_FILE.exists():
        try:
            df = pd.read_csv(DATA_FILE, encoding="utf-8")
            if not df.empty and "label" in df.columns and "message" in df.columns:
                logger.info("Loaded cached dataset: %d rows", len(df))
                return df
        except Exception as exc:
            logger.warning("Failed to read cached dataset: %s", exc)

    if download_dataset():
        return pd.read_csv(DATA_FILE, encoding="utf-8")

    logger.warning("Using bundled sample dataset (%d rows)", len(SAMPLE_DATA))
    return pd.DataFrame(SAMPLE_DATA, columns=["label", "message"])