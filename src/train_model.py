"""
Advanced Chatbot Training Module
=================================
Professional ML training with model comparison and optimization.

Author: AI Research Team
Date: December 2024
License: MIT
"""

import json
import pickle
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any
import time

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_nltk_data():
    """Download required NLTK datasets."""
    required_data = ['punkt', 'stopwords', 'punkt_tab']
    for item in required_data:
        try:
            nltk.download(item, quiet=True)
        except Exception as e:
            logger.warning(f"Could not download {item}: {e}")
    logger.info("✓ NLTK data ready")


download_nltk_data()


class AdvancedChatbotTrainer:
    """
    Advanced chatbot training with multiple ML algorithms and optimization.
    """
    
    def __init__(self, intents_file: str = 'data/intents.json'):
        """Initialize the trainer."""
        self.intents_file = intents_file
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        self.models = {}
        self.intents_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_intents(self) -> List[Dict[str, Any]]:
        """Load and validate intents data."""
        logger.info("="*70)
        logger.info("STEP 1: LOADING TRAINING DATA")
        logger.info("="*70)
        
        try:
            with open(self.intents_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.intents_data = data['intents']
            
            # Validate data
            total_patterns = sum(len(intent['patterns']) for intent in self.intents_data)
            avg_patterns = total_patterns / len(self.intents_data)
            
            logger.info(f"✓ Loaded {len(self.intents_data)} intent categories")
            logger.info(f"✓ Total training examples: {total_patterns}")
            logger.info(f"✓ Average examples per intent: {avg_patterns:.1f}")
            
            return self.intents_data
            
        except FileNotFoundError:
            logger.error(f"❌ Error: {self.intents_file} not found!")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON format: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Advanced text preprocessing."""
        try:
            tokens = nltk.word_tokenize(text.lower())
        except:
            tokens = text.lower().split()
        
        # Filter and stem
        tokens = [
            self.stemmer.stem(token)
            for token in tokens
            if token.isalpha() and token not in self.stop_words
        ]
        
        return ' '.join(tokens)
    
    def prepare_training_data(self) -> Tuple[List[str], List[str]]:
        """Prepare and preprocess training data."""
        logger.info("\n" + "="*70)
        logger.info("STEP 2: TEXT PREPROCESSING")
        logger.info("="*70)
        
        documents = []
        classes = []
        
        for intent in self.intents_data:
            for pattern in intent['patterns']:
                processed = self.preprocess_text(pattern)
                if processed:  # Only add non-empty processed text
                    documents.append(processed)
                    classes.append(intent['tag'])
        
        logger.info("✓ Tokenization completed")
        logger.info("✓ Lowercase conversion applied")
        logger.info("✓ Stop words removed")
        logger.info("✓ Stemming applied")
        logger.info(f"✓ Processed {len(documents)} training examples")
        
        return documents, classes
    
    def create_features(self, documents: List[str], classes: List[str]) -> Tuple:
        """Create TF-IDF feature vectors."""
        logger.info("\n" + "="*70)
        logger.info("STEP 3: FEATURE EXTRACTION (TF-IDF)")
        logger.info("="*70)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            documents, classes,
            test_size=0.2,
            random_state=42,
            stratify=classes
        )
        
        # Create TF-IDF features
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        self.X_train = X_train_tfidf
        self.X_test = X_test_tfidf
        self.y_train = y_train
        self.y_test = y_test
        
        logger.info(f"✓ Training set: {len(X_train)} examples")
        logger.info(f"✓ Test set: {len(X_test)} examples")
        logger.info(f"✓ Feature dimensions: {X_train_tfidf.shape[1]} features")
        logger.info(f"✓ Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
        return X_train_tfidf, X_test_tfidf, y_train, y_test
    
    def train_model(self, name: str, model, use_cv: bool = False):
        """Train and evaluate a model."""
        logger.info(f"\n{'─'*70}")
        logger.info(f"Training {name}...")
        logger.info(f"{'─'*70}")
        
        try:
            # Training
            start_time = time.time()
            model.fit(self.X_train, self.y_train)
            training_time = time.time() - start_time
            
            # Prediction
            start_pred = time.time()
            y_pred = model.predict(self.X_test)
            prediction_time = (time.time() - start_pred) / self.X_test.shape[0]
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Cross-validation (optional)
            cv_score = None
            if use_cv and self.X_train.shape[0] >= 50:
                try:
                    cv_scores = cross_val_score(
                        model, self.X_train, self.y_train,
                        cv=min(5, len(set(self.y_train))),
                        scoring='accuracy'
                    )
                    cv_score = cv_scores.mean()
                except:
                    pass
            
            self.models[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_score': cv_score,
                'training_time': training_time,
                'prediction_time': prediction_time,
                'predictions': y_pred
            }
            
            logger.info(f"✓ Training completed: {training_time:.4f}s")
            logger.info(f"✓ Test Accuracy: {accuracy*100:.2f}%")
            if cv_score:
                logger.info(f"✓ Cross-validation Score: {cv_score*100:.2f}%")
            logger.info(f"✓ Avg prediction time: {prediction_time*1000:.4f}ms")
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Error training {name}: {e}")
            raise
    
    def train_all_models(self):
        """Train multiple ML models."""
        logger.info("\n" + "="*70)
        logger.info("STEP 4: TRAINING MACHINE LEARNING MODELS")
        logger.info("="*70)
        
        # Naive Bayes
        self.train_model(
            "Naive Bayes",
            MultinomialNB(alpha=0.5)
        )
        
        # SVM
        self.train_model(
            "Support Vector Machine",
            SVC(kernel='linear', probability=True, random_state=42, C=1.0)
        )
        
        # Logistic Regression
        self.train_model(
            "Logistic Regression",
            LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        )
        
        # Random Forest
        self.train_model(
            "Random Forest",
            RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        )
    
    def compare_models(self) -> str:
        """Compare and display model performance."""
        logger.info("\n" + "="*70)
        logger.info("STEP 5: MODEL COMPARISON & EVALUATION")
        logger.info("="*70)
        
        print(f"\n{'Model':<25} {'Accuracy':<12} {'Train Time':<15} {'Pred Time':<15}")
        print("─"*70)
        
        for name, info in self.models.items():
            cv_str = f"(CV: {info['cv_score']*100:.1f}%)" if info['cv_score'] else ""
            print(f"{name:<25} {info['accuracy']*100:>6.2f}%      "
                  f"{info['training_time']:>8.4f}s      "
                  f"{info['prediction_time']*1000:>8.4f}ms")
        
        # Find best model
        best_model = max(self.models.items(), key=lambda x: x[1]['accuracy'])
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🏆 BEST MODEL: {best_model[0]}")
        logger.info(f"   Accuracy: {best_model[1]['accuracy']*100:.2f}%")
        logger.info(f"   Speed: {best_model[1]['prediction_time']*1000:.2f}ms")
        logger.info(f"{'='*70}")
        
        return best_model[0]
    
    def save_model(self, model_name: str = 'Naive Bayes'):
        """Save the trained model."""
        logger.info("\n" + "="*70)
        logger.info("STEP 6: SAVING MODEL")
        logger.info("="*70)
        
        model_data = {
            'model': self.models[model_name]['model'],
            'vectorizer': self.vectorizer,
            'intents': self.intents_data,
            'stemmer': self.stemmer,
            'stop_words': self.stop_words,
            'metadata': {
                'model_type': model_name,
                'accuracy': self.models[model_name]['accuracy'],
                'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'num_intents': len(self.intents_data),
                'vocab_size': len(self.vectorizer.vocabulary_)
            }
        }
        
        # Save to src directory
        filepath = Path('src/chatbot_model.pkl')
        filepath.parent.mkdir(exist_ok=True)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"✓ Model saved: {filepath}")
            logger.info(f"✓ Model type: {model_name}")
            logger.info(f"✓ Accuracy: {self.models[model_name]['accuracy']*100:.2f}%")
            logger.info(f"✓ File size: {filepath.stat().st_size / 1024:.2f} KB")
            
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")
            raise
    
    def train_pipeline(self):
        """Execute complete training pipeline."""
        try:
            start_time = time.time()
            
            logger.info("\n" + "="*70)
            logger.info("🚀 CHATBOT TRAINING PIPELINE STARTED")
            logger.info("="*70)
            
            # Steps
            self.load_intents()
            documents, classes = self.prepare_training_data()
            self.create_features(documents, classes)
            self.train_all_models()
            best_model = self.compare_models()
            self.save_model('Naive Bayes')  # Save Naive Bayes for speed
            
            total_time = time.time() - start_time
            
            logger.info("\n" + "="*70)
            logger.info("✅ TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("="*70)
            logger.info(f"⏱️  Total time: {total_time:.2f} seconds")
            logger.info(f"📊 Best model: {best_model}")
            logger.info(f"🎯 Ready for deployment!")
            logger.info("\n💡 Next step: Run 'streamlit run src/app.py'")
            logger.info("="*70)
            
            return True
            
        except Exception as e:
            logger.error(f"\n❌ Training failed: {e}")
            return False


def main():
    """Main training function."""
    trainer = AdvancedChatbotTrainer('data/intents.json')
    success = trainer.train_pipeline()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
