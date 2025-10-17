# Step 1: Import necessary libraries
import spacy
import pandas as pd
from spacy import displacy
import random
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

print("Starting Amazon Reviews NLP Analysis...")

# Step 2: Load spaCy model
print("\n=== STEP 1: Loading spaCy Model ===")
try:
    # Try loading the medium English model for better NER performance
    nlp = spacy.load("en_core_web_md")
    print("✅ Loaded en_core_web_md model")
except OSError:
    print("⚠️ Medium model not found. Installing...")
    # If medium model is not available, use the small one and download if needed
    try:
        # Prefer using spaCy's Python API to download the model
        from spacy.cli import download as spacy_download
        spacy_download("en_core_web_sm")
    except Exception as e:
        # Fallback to invoking the spacy download via subprocess if needed
        import subprocess, sys
        print(f"spaCy CLI download failed with: {e}; attempting subprocess call...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")
    print("✅ Loaded en_core_web_sm model")

print("Model loaded successfully!")
print(f"Pipeline components: {nlp.pipe_names}")

# Step 3: Create sample Amazon product reviews data
print("\n=== STEP 2: Creating Sample Amazon Reviews Data ===")
amazon_reviews = [
    "I absolutely love my new iPhone 14 Pro from Apple! The camera quality is amazing and battery life lasts all day.",
    "The Samsung Galaxy S23 Ultra is disappointing. The battery drains quickly and the screen has issues.",
    "Bought this Sony WH-1000XM4 headphones and they're fantastic! Noise cancellation is incredible.",
    "The Nike Air Max shoes are uncomfortable and started falling apart after two weeks. Very poor quality.",
    "Amazon Echo Dot 4th Gen is a great smart speaker for the price. Alexa works perfectly with my smart home.",
    "The Dell XPS 13 laptop has excellent performance but the keyboard is terrible for typing.",
    "Microsoft Surface Pro 9 is the perfect tablet for work and creativity. Highly recommend!",
    "This Adidas running shoes are the worst purchase I've made. They hurt my feet and the sole came off.",
    "Google Pixel 7 Pro has an outstanding camera but the battery life could be better.",
    "The Bose QuietComfort headphones are worth every penny. Best sound quality I've ever experienced.",
    "HP Pavilion laptop stopped working after one month. Customer service from HP was horrible.",
    "I'm very happy with my new MacBook Air from Apple. The M2 chip is incredibly fast.",
    "The Lenovo ThinkPad is a reliable workhorse for business use. Keyboard is comfortable for long typing sessions.",
    "This Asus monitor has dead pixels right out of the box. Very disappointed with Asus quality control.",
    "PlayStation 5 from Sony is amazing for gaming! The graphics and loading times are impressive."
]

# Create a DataFrame for better organization
reviews_df = pd.DataFrame({
    'review_text': amazon_reviews,
    'review_id': range(1, len(amazon_reviews) + 1)
})

print("Sample Reviews DataFrame:")
print(reviews_df.head())
print(f"\nTotal reviews: {len(reviews_df)}")

# Step 4: Perform Named Entity Recognition (NER)
print("\n=== STEP 3: Named Entity Recognition (NER) ===")
print("Extracting product names and brands from reviews...")

def extract_entities(review_text):
    """Extract entities from a single review using spaCy NER"""
    doc = nlp(review_text)
    entities = []
    
    for ent in doc.ents:
        # Focus on product-related entities (ORG, PRODUCT, and sometimes GPE for brands)
        if ent.label_ in ['ORG', 'PRODUCT', 'GPE']:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
    
    return entities

# Apply NER to all reviews
reviews_df['entities'] = reviews_df['review_text'].apply(extract_entities)

print("\nNER Results for first 3 reviews:")
for i, (review, entities) in enumerate(zip(reviews_df['review_text'][:3], reviews_df['entities'][:3])):
    print(f"\nReview {i+1}: {review}")
    if entities:
        for entity in entities:
            print(f"  - {entity['text']} ({entity['label']})")
    else:
        print("  No relevant entities found")

# Step 5: Analyze entity statistics
print("\n=== STEP 4: Entity Analysis ===")
all_entities = []
for entities in reviews_df['entities']:
    all_entities.extend(entities)

if all_entities:
    # Count entity types
    entity_types = Counter([entity['label'] for entity in all_entities])
    print("Entity types distribution:")
    for entity_type, count in entity_types.items():
        print(f"  {entity_type}: {count}")
    
    # Count most common entities
    entity_texts = [entity['text'] for entity in all_entities]
    common_entities = Counter(entity_texts).most_common(10)
    print("\nMost common entities:")
    for entity, count in common_entities:
        print(f"  {entity}: {count}")
else:
    print("No entities found in the reviews")

# Step 6: Rule-based sentiment analysis
print("\n=== STEP 5: Rule-Based Sentiment Analysis ===")
# Define sentiment words and their weights
positive_words = {
    'love', 'amazing', 'fantastic', 'great', 'excellent', 'perfect', 'outstanding',
    'incredible', 'awesome', 'happy', 'good', 'best', 'worth', 'recommend', 'fast',
    'comfortable', 'reliable', 'impressive', 'fantastic', 'perfectly'
}

negative_words = {
    'disappointing', 'terrible', 'worst', 'horrible', 'poor', 'uncomfortable',
    'disappointed', 'bad', 'issues', 'drains', 'falling', 'hurt', 'stopped'
}

def analyze_sentiment(text):
    """Perform rule-based sentiment analysis"""
    doc = nlp(text.lower())
    positive_score = 0
    negative_score = 0
    
    # Check for positive words
    for token in doc:
        if token.text in positive_words:
            positive_score += 1
        elif token.text in negative_words:
            negative_score += 1
    
    # Determine sentiment based on scores
    if positive_score > negative_score:
        return 'positive'
    elif negative_score > positive_score:
        return 'negative'
    else:
        return 'neutral'

# Apply sentiment analysis to all reviews
reviews_df['sentiment'] = reviews_df['review_text'].apply(analyze_sentiment)

print("Sentiment analysis completed!")
print("\nSentiment distribution:")
sentiment_counts = reviews_df['sentiment'].value_counts()
for sentiment, count in sentiment_counts.items():
    print(f"  {sentiment}: {count} reviews")

# Step 7: Combine NER and sentiment results
print("\n=== STEP 6: Combined NER and Sentiment Results ===")
# Create a summary of products/brands and their associated sentiments
product_sentiments = []

for _, row in reviews_df.iterrows():
    entities = row['entities']
    sentiment = row['sentiment']
    
    if entities:
        for entity in entities:
            if entity['label'] in ['ORG', 'PRODUCT']:
                product_sentiments.append({
                    'brand_product': entity['text'],
                    'sentiment': sentiment,
                    'review': row['review_text'][:50] + "..."  # Preview
                })

# Create DataFrame for product sentiments
if product_sentiments:
    product_df = pd.DataFrame(product_sentiments)
    print("\nProduct/Brand Sentiment Analysis:")
    print(product_df.head(10))
    
    # Analyze sentiment by brand/product
    print("\nSentiment by Brand/Product:")
    sentiment_by_brand = product_df.groupby(['brand_product', 'sentiment']).size().unstack(fill_value=0)
    print(sentiment_by_brand)
else:
    print("No product/brand entities found for sentiment analysis")

# Step 8: Visualize the results
print("\n=== STEP 7: Visualization ===")
plt.figure(figsize=(15, 10))

# Plot 1: Sentiment distribution
plt.subplot(2, 2, 1)
sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])
plt.title('Review Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=0)

# Plot 2: Entity type distribution
plt.subplot(2, 2, 2)
if all_entities:
    entity_type_counts = pd.Series(entity_types)
    entity_type_counts.plot(kind='bar', color='skyblue')
    plt.title('Entity Types Distribution')
    plt.xlabel('Entity Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)

# Plot 3: Top entities
plt.subplot(2, 2, 3)
if all_entities:
    top_entities = pd.Series(dict(common_entities[:7]))
    top_entities.plot(kind='barh', color='lightcoral')
    plt.title('Top 7 Most Mentioned Entities')
    plt.xlabel('Count')

# Plot 4: Sentiment by brand (if we have data)
plt.subplot(2, 2, 4)
if product_sentiments:
    # Get top 5 brands by mention count
    top_brands = product_df['brand_product'].value_counts().head(5).index
    top_brands_data = product_df[product_df['brand_product'].isin(top_brands)]
    
    if not top_brands_data.empty:
        sentiment_pivot = top_brands_data.groupby(['brand_product', 'sentiment']).size().unstack(fill_value=0)
        sentiment_pivot.plot(kind='bar', ax=plt.gca(), color=['red', 'gray', 'green'])
        plt.title('Sentiment for Top 5 Brands/Products')
        plt.xlabel('Brand/Product')
        plt.ylabel('Number of Reviews')
        plt.xticks(rotation=45)
        plt.legend(title='Sentiment')

plt.tight_layout()
plt.show()

# Step 9: Display detailed NER visualization for sample reviews
print("\n=== STEP 8: NER Visualization ===")
print("Visualizing entities in sample reviews...")

# Select 2 sample reviews for detailed NER visualization
sample_indices = [0, 2]  # First and third review
for idx in sample_indices:
    review_text = reviews_df.iloc[idx]['review_text']
    doc = nlp(review_text)
    
    print(f"\nReview {idx + 1}: {review_text}")
    print("Entities found:")
    for ent in doc.ents:
        print(f"  - {ent.text} ({ent.label_})")
    
    # You can also use displacy for visual rendering (works better in Jupyter)
    # print("\nVisual NER representation:")
    # displacy.render(doc, style="ent", jupyter=True)

# Step 10: Final results summary
print("\n=== STEP 9: Final Results Summary ===")
print("📊 NAMED ENTITY RECOGNITION RESULTS:")
print(f"   - Total entities extracted: {len(all_entities)}")
print(f"   - Unique brands/products: {len(set([e['text'] for e in all_entities]))}")
print(f"   - Most common entity types: {', '.join([f'{k} ({v})' for k, v in entity_types.most_common(3)])}")

print("\n📊 SENTIMENT ANALYSIS RESULTS:")
print(f"   - Positive reviews: {sentiment_counts.get('positive', 0)}")
print(f"   - Negative reviews: {sentiment_counts.get('negative', 0)}")
print(f"   - Neutral reviews: {sentiment_counts.get('neutral', 0)}")
print(f"   - Overall sentiment ratio: {sentiment_counts.get('positive', 0)/len(reviews_df)*100:.1f}% positive")

print("\n🎯 KEY INSIGHTS:")
# Find brands with mixed sentiments
if product_sentiments:
    brand_sentiments = product_df.groupby('brand_product')['sentiment'].apply(lambda x: x.mode()[0] if not x.mode().empty else 'neutral')
    positive_brands = brand_sentiments[brand_sentiments == 'positive'].index.tolist()
    negative_brands = brand_sentiments[brand_sentiments == 'negative'].index.tolist()
    
    if positive_brands:
        print(f"   - Generally positive brands: {', '.join(positive_brands[:3])}")
    if negative_brands:
        print(f"   - Generally negative brands: {', '.join(negative_brands[:3])}")

print("✅ Named Entity Recognition performed to extract product names and brands")
print("✅ Rule-based sentiment analysis implemented")
