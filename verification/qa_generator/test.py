from url2text import extract_text_from_url


url = "https://www.facebook.com/DamascusArtistss/posts/1719544955991034?rdid=Is1uA49NzWud25eR#"
print("\n🔍 Extracting text:")
extracted_text = extract_text_from_url(url)
with open("extracted_text.txt", "w", encoding="utf-8") as f:
    f.write(extracted_text)
print("✓ Extracted text saved to extracted_text.txt.")
