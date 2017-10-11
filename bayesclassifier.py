import nltk

class NaiveBayesClassifier:
    
    def __init__(self, categories):
        self.categories = categories
        self.word_memo = {} # a 2-dimensional [string=>[string=>int]] hashmap storing the number of documents in each category each word appears in
        self.document_count = {} # a 1-dimensional [string=>int] hashmap storing the number of training documents belonging to each category
        self.total_document_count = 0
        for category in categories:
            self.word_memo[category] = {} 
            self.document_count[category] = 0 

    '''
        returns total word count for category
        (returns total word count for all categories if input by default)
    '''
    def total_word_count(self, category=None):
        if not category:
            word_count = 0
            for category in self.categories:
                for word in self.word_memo[category]:
                    word_count += self.word_memo[category][word]
            return word_count
        return sum(self.word_memo[category][word] for word in self.word_memo[category])
        
    '''
        returns the number of unique words/tokens in the entire corpus
    '''
    def vocab_size(self, category=None):
        if not category:
            vocab_set = set([])
            for cat in self.categories:
                for word in self.word_memo[cat]:
                    vocab_set.add(word)
            return len(vocab_set)
        return len([word for word in self.word_memo[category]])
    
    '''
        returns the amount of documents in the given category the word appears in
    '''
    def word_count(self, word, category):
        if not word in self.word_memo[category]:
            return 0
        return self.word_memo[category][word]
    
    '''
        returns the most likely category for the given text input, along with the probability
    '''
    def classify(self, text):
        bag_of_words = self.tokenize(text)
        max_likelihood = 0
        max_category = self.categories[0]
        for category in self.categories:
            likelihood = self.text_likelihood(bag_of_words, category)
            if likelihood > max_likelihood:
                max_likelihood = likelihood
                max_category = category
        return [max_category, max_likelihood]
        
            
    '''
        returns p(category)
    '''
    def category_probability(self, category):
        return self.document_count[category] / sum(self.document_count[cat] for cat in self.categories)
   
    '''
        returns p(word|category)
    '''
    def word_probability_category(self, word, category):
        if self.document_count[category] > 0:
            return self.word_count(word, category) / self.document_count[category]
        return 0
    
    '''
        returns p(word)
    '''
    def word_probability(self, word):
        return sum(self.word_count(word, cat) for cat in self.categories) / self.total_document_count
        
    '''
        returns a likelihood score for the given text input and category
    '''
    def text_likelihood(self, bag_of_words, category):
        score = 1
        
        for word in bag_of_words:
            word_prob = self.word_probability(word)
            word_cat_probability = self.word_probability_category(word, category)
            if word_prob > 0 and word_cat_probability > 0:
               
                    score *= word_cat_probability / word_prob
        return score * self.category_probability(category)
    
    ''' 
        given a document and a category as input,
        tokenize the document and increment the count of
        each word for the category in the word memo
    '''
    def train(self, text, category):
        self.document_count[category] += 1
        self.total_document_count += 1
        bag_of_words = self.tokenize(text)
        for word in bag_of_words:
            self.add_word(word, category)
    
    '''
        add a word (from training input) to the word map
    '''
    def add_word(self, word, category):
        if not word in self.word_memo[category]:
            self.word_memo[category][word] = 1
        else:
            self.word_memo[category][word] += 1
            
    '''
        remove a word (from training input) to the word map
    '''
    def remove_word(self, word, category=None):
        if category:
            self.word_memo[category].pop(word, None)
        else:
            for cat in self.categories:
                self.word_memo[cat].pop(word, None)
           

    '''
        accepts raw text input and returns a bag of words, with stopwords removed.
        Uses NLTK library for tokenization and stopword corpus
    '''
    def tokenize(self, text):
        stopwords = set(nltk.corpus.stopwords.words('english'))
        words = nltk.word_tokenize(text)
        return list(set([word for word in words if not word in stopwords]))
        
        
    '''
        removes all words from memo if p(word) >= threshold
        i.e., removing the most common (noise) words from the corpus
    '''
    def cleanup(self, threshold=0.2):
        removed_count = 0
        for cat in self.categories:
            to_remove = set([])
            for word in self.word_memo[cat]:
                if self.word_probability(word) >= threshold:
                    to_remove.add(word)
                    
            for word in to_remove:
                self.remove_word(word)
                removed_count += 1
                
        print("cleanup removed " + str(removed_count) + " words")