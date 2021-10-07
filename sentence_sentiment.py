import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
# sentences = ["VADER is smart, handsome, and funny.", # positive sentence example
#    "VADER is smart, handsome, and funny!", # punctuation emphasis handled correctly (sentiment intensity adjusted)
#    "VADER is very smart, handsome, and funny.",  # booster words handled correctly (sentiment intensity adjusted)
#    "VADER is VERY SMART, handsome, and FUNNY.",  # emphasis for ALLCAPS handled
#    "VADER is VERY SMART, handsome, and FUNNY!!!",# combination of signals - VADER appropriately adjusts intensity
#    "VADER is VERY SMART, really handsome, and INCREDIBLY FUNNY!!!",# booster words & punctuation make this close to ceiling for score
#    "The book was good.",         # positive sentence
#    "The book was kind of good.", # qualified positive sentence is handled correctly (intensity adjusted)
#    "The plot was good, but the characters are uncompelling and the dialog is not great.", # mixed negation sentence
#    "A really bad, horrible book.",       # negative sentence with booster words
#    "At least it isn't a horrible book.", # negated negative sentence with contraction
#    ":) and :D",     # emoticons handled
#    "",              # an empty string is correctly handled
#    "Today sux",     #  negative slang handled
#    "Today sux!",    #  negative slang with punctuation emphasis handled
#    "Today SUX!",    #  negative slang with capitalization emphasis
#    "Today kinda sux! But I'll get by, lol" # mixed sentiment example with slang and constrastive conjunction "but"
# ]
# tricky_sentences = [
#    "Most automated sentiment analysis tools are shit.",
#    "VADER sentiment analysis is the shit.",
#    "Sentiment analysis has never been good.",
#    "Sentiment analysis with VADER has never been this good.",
#    "Warren Beatty has never been so entertaining.",
#    "I won't say that the movie is astounding and I wouldn't claim that the movie is too banal either.",
#    "I like to hate Michael Bay films, but I couldn't fault this one",
#    "It's one thing to watch an Uwe Boll film, but another thing entirely to pay for it",
#    "The movie was too good",
#    "usually around the time the 90 day warranty expires",
#    "the twin towers collapsed today"
# ]
# sentences.extend(tricky_sentences)
sentences = ["The lungs are clear .",
"there is no focal consolidation , pleural effusion , or pneumothorax .",
"the heart and mediastinum are normal size and shape .",
"xxxx and soft tissues are unremarkable ."]

# sentences2 = ["The heart and lungs have xxxx xxxx in the interval .",
# "both lungs are clear and expanded .",
# "heart and mediastinum normal ."]
sentences2 = ["The lungs are not clear .",
"there is focal consolidation , pleural effusion , or pneumothorax .",
"the heart and mediastinum are normal size and shape .",
"xxxx and soft tissues are unremarkable ."]

sid = SentimentIntensityAnalyzer()
print("=======================sentence=========")
for sentence in sentences:
    print(sentence)
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end='')
    print()
print("=======================sentence2=========")
for sentence in sentences2:
    print(sentence)
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end='')
    print()