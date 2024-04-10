from wordcloud import WordCloud
#from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
#from nltk.stem import WordNetLemmatizer
import nltk
import matplotlib.pyplot as plt

# Download NLTK resources for part-of-speech tagging
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


"""
Visualising data in our legal text documents. One of the visualisation methods with respect to textual data is a word cloud.
In our word cloud we choose to first preprocess our data and conduct a Part-of-Speech tagging.
Thereafter we chose to generate a noun Word Cloud and a verb Word Cloud as we feel that these parts of speech 
would provide important information within the legal text.

Nouns: Nouns often represent the key entities, concepts, and objects within the legal text. 
       This includes names of parties involved, places, legal terminology, and other substantive terms. 
       Nouns can provide a good overview of the main subjects covered in the document.

Verbs: Verbs typically represent actions, processes, or states of being. 
       In a legal text, verbs can indicate what actions are being taken, 
    what is being requested or mandated, or what happened in a particular case. 
    Verbs can help identify the activities and events described within the document.
"""

# Extract the verbs
def extract_verbs(text):
    verbs = []
    # Stopword removal
    pos_tag_tokens = pos_tag(word_tokenize(text))
    
    verb_l = [ele[0] for ele in pos_tag_tokens if ele[1] == 'VB' or ele[1] == 'VBP' or ele[1] == 'VBZ' or ele[1] == 'VBN' or ele[1] == 'VBD' or ele[1] == 'VBG']
    
    #print(pos_tag_tokens)
    verbs.extend(verb_l)
    #print(adjectives)
    
    return verbs

def extract_nouns(text):
    nouns = []
    
    pos_tag_tokens = pos_tag(word_tokenize(text))
    
    noun_l = [ele[0] for ele in pos_tag_tokens if ele[1] == 'NN']
    
    #print(pos_tag_tokens)
    nouns.extend(noun_l)
    #print(adjectives)
    
    return nouns


def generate_word_cloud_adjectives(text, title):
    text = ' '.join(extract_verbs(text))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    print(text)
    # Plot the Word Cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud)
    plt.title(title)
    plt.axis('off')

def generate_word_cloud_nouns(text, title):
    text = ' '.join(extract_nouns(text))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    print(text)
    # Plot the Word Cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud)
    plt.title(title)
    plt.axis('off')

#CSV



text = """
Bali and Lombok are neighbouring islands; both are part of the Indonesian archipelago. It is easy to appreciate each island as an attractive tourist destination – majestic scenery; rich culture; white sands and warm, azure waters draw visitors like magnets every year. Snorkelling and diving around the nearby Gili Islands is magnificent, with marine fish, starfish, turtles and coral reef present in abundance. Whereas Bali is predominantly a Hindu country, the inhabitants of Lombok are mostly Muslim with a Hindu minority. Bali is known for its elaborate, traditional dancing which is inspired by its Hindi beliefs. Most of the dancing portrays tales of good versus evil; to watch it is a breathtaking experience. Art is another Balinese passion – batik paintings and carved statues make popular souvenirs. Artists can be seen whittling and painting on the streets, particularly in Ubud. The island is home to some spectacular temples, the most significant being the Mother Temple, Besakih. Lombok, too, has some impressive points of interest – the majestic Gunung Rinjani is an active volcano and the second highest peak in Indonesia. Like Bali, Lombok has several temples worthy of a visit, though they are less prolific. Lombok remains the most understated of the two islands."""
# Test
#print(extract_verbs(text))
generate_word_cloud_adjectives(text, 'Verb wordcloud')
generate_word_cloud_nouns(text, 'Noun wordcloud')
plt.show()

""" 
PROCESS

.join all the facts/issues in a column (for a particular area of law (optional)) 
Run both of these functions to generate the word cloud for nouns and adjectives.
"""