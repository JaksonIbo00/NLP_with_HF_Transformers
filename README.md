<h1 align="center"> Natural Language Processing with Hugging Face Transformers </h1>

## NAME : JAKSON ROMARIO SIMSON IBO

## MY TODO :

## Exercise 1 - Sentiment Analysis
```python
# TODO 1:
classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
classifier("Having three long haired, heavy shedding dogs at home, I was pretty skeptical that this could hold up to all the hair and dirt they trek in, but this wonderful piece of tech has been nothing short of a godsend for me! ")

# TODO 2:
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
classifier("Having three long haired, heavy shedding dogs at home, I was pretty skeptical that this could hold up to all the hair and dirt they trek in, but this wonderful piece of tech has been nothing short of a godsend for me! ")
```

### Result :
```python
# TODO 1:
[{'label': 'LABEL_2', 'score': 0.6510636806488037}]

# TODO 2:
[{'label': 'POSITIVE', 'score': 0.9982457160949707}]
```

### Analysis on Example 1 :
Dari kedua hasil analisis sentimen, dapat dilihat bahwa untuk model "distilbert-base-uncased-finetuned-sst-2-english" menghasilkan skor kepercayaan yang lebih tinggi bila dibandingkan dengan model "cardiffnlp/twitter-roberta-base-sentiment" yang menandakan bahwa memiliki tingkat keyakinan yang lebih kuat untuk mendeteksi emosi positif dari teks. Selain itu, model "cardiffnlp/twitter-roberta-base-sentiment" menggunakan label 0, 1, dan 2 untuk menentukan kategori output/keluarannya, dimana 0 = negatif, 1 = netral, dan 2 = positif. Dari contoh diatas dapat dilihat bahwa kedua model menghasilkan output positif.

## Exercise 2 - Topic Classification
```python
# TODO
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier(
    "I am from the East side of Indonesia, I am Papuan",
    candidate_labels=["art", "music", "ethnicity"],
)
```

### Result :
```python
{'sequence': 'I am from the East side of Indonesia, I am Papuan',
 'labels': ['ethnicity', 'art', 'music'],
 'scores': [0.9723653197288513, 0.017297690734267235, 0.01033700630068779]}
```

### Analysis on Example 2 :
Model diatas mengklasifikasikan kalimat kedalam beberapa kelas/kategori yang telah dipilih. Pada contoh diatas, terdapat 3 kelas yaitu ethnicity, art, dan music dimana ethnicity memiliki skor paling tinggi yang membuktikan bahwa kelas ethnicity merupakan topik yang dibahas dalam kalimat.

## Exercise 3 - Text Generation Models
``` python
# TODO
generator = pipeline('text-generation', model = 'gpt2')
generator("It is very hot in here, I want to", max_length = 35, num_return_sequences=5)
```

### Result :
``` python
[{'generated_text': 'It is very hot in here, I want to burn."\n\nThat\'s not good when you\'re trying to save money on pizza or pizza-making. It\'s like'},
 {'generated_text': 'It is very hot in here, I want to run around, I need to be able to breathe air. I always thought you need an air-conditioning system," he'},
 {'generated_text': "It is very hot in here, I want to be in it. If you know what you're going to do I'm going to pull all this pressure. In the offseason"},
 {'generated_text': 'It is very hot in here, I want to do something and not have to put you next to someone and have nothing bad to do with it right at the moment."\n'},
 {'generated_text': 'It is very hot in here, I want to go and have some of those drinks. You know, we got to go out there and show our support to the people that'}]
```

### Analysis on Example 3 :
Teknik NLP ini mampu menyelesaikan kalimat-kalimat yang belum selesai atau finish menajdi sebuah kalimat dengan gpt2 sebagai text generator. Pada contoh diatas, kalimat yang diberikan adalah "it is very hot in here, i want to", sebuah kalimat yang belum selesai kemudian diselesaikan dengan beberapa opsi pilihan menjadi sebuah kalimat utuh.

## Exercise 4 - Name Entity Recognition
``` python
# TODO
nlp = pipeline("ner", model="Jean-Baptiste/camembert-ner", grouped_entities=True)
example = "My friend John works at Google Office in Jakarta."
ner_results = nlp(example)
print(ner_results)
```

### Result :
``` python
[{'entity_group': 'PER', 'score': np.float32(0.72509813), 'word': 'John', 'start': 9, 'end': 14}, {'entity_group': 'ORG', 'score': np.float32(0.92203903), 'word': 'Google Office', 'start': 23, 'end': 37}, {'entity_group': 'LOC', 'score': np.float32(0.9984701), 'word': 'Jakarta', 'start': 40, 'end': 48}]
```

### Analysis on Example 4 :
Pada teknik NLP ini, sebuah kalimat yang dibuat dengan berisi nama, lokasi, dan organisasi akan langsung diklasifikasikan oleh model, beserta dengan skor, dan index huruf awal kata serta akhir kata. Pada contoh diatas, John berhasil dikategorikan sebagai nama, Google office sebagai organisasi, dan Jakarta sebagai lokasi.

## Exercise 5 - Question Answering
``` python
# TODO
question_answerer = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question_answerer(
    question="What is the smallest planet in our solar system?",
    context="Mercury is the closest planet to the Sun in our solar system. It is the smallest planet and has a rocky surface filled with many craters. Because Mercury is very close to the Sun, the temperature can become extremely hot during the day and very cold at night. The planet moves around the Sun faster than any other planet and does not have any moons."
)
```

### Result :
``` python
{'score': 0.9145805835723877, 'start': 0, 'end': 7, 'answer': 'Mercury'}
```

### Analysis on Example 5 :
Model diatas bekerja ketika pengguna memasukan sebuah informasi, kemudian memberikan sebuah pertanyaan yang berkaitan dengan informasi yang telah disampaikan. Model akan menjawab pertanyaan tersebut sesuai dengan informasi yang telah diberikan. Pada contoh diatas, pertanyaan apakah merkurius merupakan planet terkecil berhasil terjawab berdasarkan informasi yang telah diberikan dengan teknik question answering ini.

## Exercise 6 - Text Summarization
``` python
# TODO
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6",  max_length=59)
summarizer(
    """
Antarctica is the coldest and driest continent on Earth, located at the southernmost part of the planet.
Almost all of Antarctica is covered by a thick layer of ice that stores most of the world’s fresh water.
The continent experiences extremely low temperatures, strong winds, and long periods of darkness during winter.
Even though the environment is very harsh, many animals such as penguins, seals, and whales can survive there.
Antarctica does not have a permanent human population, but scientists from different countries stay there temporarily to conduct research about climate, ice, oceans, and wildlife.
The continent is very important for understanding global climate change because changes in Antarctic ice can affect sea levels around the world.
"""
)
```

### Result :
``` python
[{'summary_text': ' Antarctica is the coldest and driest continent on Earth, located at the southernmost part of the planet . Almost all of Antarctica is covered by a thick layer of ice that stores most of the world’s fresh water . The continent is very important for understanding global climate change'}]
```

### Analysis on Example 6 :
Teknik ini merangkum/meringkas sebuah informasi utuh yang diberikan menjadi sebuah paragraf ringkas yang jelas tanpa mengubah kalimat yang ada didalam paragraf sebelumnya. Teknik ini mengambil informasi-informasi penting yang ada di dalam paragraf awal kemudian mengubahnya menjadi paragraf baru yang lebih ringkas. Pada contoh diatas model yang digunakan adalah "sshleifer/distilbart-cnn-12-6" yang mengubah paragraf utuh sebelumnya menjadi paragraf ringkas dengan panjang maksimal 59 kata.

## Exercise 7 - Translation
``` python
# TODO
translator = pipeline("translation_en_to_de", model="t5-small")
print(translator("I have a cold", max_length=40))
```

### Result :
``` python
[{'translation_text': 'Ich habe eine Kälte'}]
```

### Analysis on Example 7 :
Teknik ini digunakan untuk mengubah sebuah bahasa menjadi bahasa lainnya sesuai keinginan pengguna. Teknik ini digunakan dalam teknologi seperti Google Translate untuk mengubah berbagai bahasa yang ada di dunia sesuai keinginan pengguna. Pada contoh diatas, model "t5-small" yang dibuat oleh google digunaka untuk mengubah bahasa inggris menjadi bahasa jerman
## 
## Analysis on this Project
Secara keseluruhan, Naturan Language Processing dengan Hugging Face Transformers ini memberikan gambaran bagaimana AI dapat dimanfaatkan untuk mengerjakan pekerjaan-pekerjaan yang berhubungan dengan bahasa manusia. Mulai dari menilai sebuah kalimat, mengklasifikasikan kata, melengkapi kalimat, sampai mengubah kalimat menjadi bahasa-bahasa yang ada di seluruh dunia. Hal ini tentu saja sangat berguna dan mempermudah berbagai tugas manusia, sehingga pekerjaan manusia dapat dikerjakan dengan lebih efektif dan efisien.



