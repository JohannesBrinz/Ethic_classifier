import pandas as pd
import re

sms_spam = pd.read_csv('SMSSpamCollection', sep='\t',
header=None, names=['Label', 'SMS'])    #read data file

print("Shape of loaded data:  ", sms_spam.shape)
sms_spam.head()         #returns first five rows

print(sms_spam['Label'].value_counts(normalize=True))           #returns a series (two) counts of unique values (ham, spam)

########      training and test set     #########
#randomize data set
data_randomized = sms_spam.sample(frac = 1, random_state=1)

#calculate indey for split
training_test_index = round(len(data_randomized) * 0.8)     #80% of data used for training

#Split into training and test sets
training_set = data_randomized[:training_test_index].reset_index(drop=True)
test_set = data_randomized[training_test_index:].reset_index(drop=True)

print("\nTraining set:\n", training_set)
print("Composition:\n", training_set['Label'].value_counts(normalize=True), "\n")
print("\nTest set:\n", test_set)
print("Composition:\n", test_set['Label'].value_counts(normalize=True), "\n")


########   data cleaning  ########
#remove upper case and punctuation
training_set['SMS'] = training_set['SMS'].str.replace(
   '\W', ' ') # Removes punctuation
training_set['SMS'] = training_set['SMS'].str.lower()

#creating the vocabulary
training_set["SMS"] = training_set["SMS"].str.split()           #splits the SMSs into Lists of words

vocabulary = []
for sms in training_set["SMS"]:
    for word in sms:
        vocabulary.append(word)

vocabulary = list(set(vocabulary))           #transforms list into a set to remove duplicates and then back to list

#creating word count list
word_count_per_sms = {unique_word: [0] * len(training_set['SMS']) for unique_word in vocabulary}    #creates empty dictionary od lists

for index, sms in enumerate(training_set['SMS']):
   for word in sms:
       word_count_per_sms[word][index] += 1

word_counts = pd.DataFrame(word_count_per_sms)         #transform to data frame
training_set_clean = pd.concat([training_set, word_counts], axis=1)     #add label and sms from training_set
print(training_set_clean.head())


###############    spam filter    ###############

# Isolating spam and ham messages first
spam_messages = training_set_clean[training_set_clean['Label'] == 'spam']
ham_messages = training_set_clean[training_set_clean['Label'] == 'ham']

# P(Spam) and P(Ham)
p_spam = len(spam_messages) / len(training_set_clean)
p_ham = len(ham_messages) / len(training_set_clean)

# N_Spam
n_words_per_spam_message = spam_messages['SMS'].apply(len)
n_spam = n_words_per_spam_message.sum()

# N_Ham
n_words_per_ham_message = ham_messages['SMS'].apply(len)
n_ham = n_words_per_ham_message.sum()

# N_Vocabulary
n_vocabulary = len(vocabulary)

# Laplace smoothing
alpha = 1


#calculating parameters
# Initiate parameters
parameters_spam = {unique_word:0 for unique_word in vocabulary}
parameters_ham = {unique_word:0 for unique_word in vocabulary}

# Calculate parameters
for word in vocabulary:
   n_word_given_spam = spam_messages[word].sum() # spam_messages already defined
   p_word_given_spam = (n_word_given_spam + alpha) / (n_spam + alpha*n_vocabulary)
   parameters_spam[word] = p_word_given_spam

   n_word_given_ham = ham_messages[word].sum() # ham_messages already defined
   p_word_given_ham = (n_word_given_ham + alpha) / (n_ham + alpha*n_vocabulary)
   parameters_ham[word] = p_word_given_ham

#define clasifier function using the naive baysian classifier
def classify(message):
   '''
   message: a string
   '''

   message = re.sub('\W', ' ', message)
   message = message.lower().split()

   p_spam_given_message = p_spam
   p_ham_given_message = p_ham

   for word in message:
      if word in parameters_spam:
         p_spam_given_message *= parameters_spam[word]

      if word in parameters_ham:
         p_ham_given_message *= parameters_ham[word]

   print('P(Spam|message):', p_spam_given_message)
   print('P(Ham|message):', p_ham_given_message)

   if p_ham_given_message > p_spam_given_message:
      print('Label: Ham')
   elif p_ham_given_message < p_spam_given_message:
      print('Label: Spam')
   else:
      print('Equal proabilities, have a human classify this!')

#test classifier
classify('Winner winner chicken dinner')


#(https://www.kdnuggets.com/2020/07/spam-filter-python-naive-bayes-scratch.html)
