# Humour.ai

I have seen a lot of people do cool projects in Computer Vision (the more hyped one) but hardly I have ever seen something good in NLP. After learning about transformers, I thought I should do something in NLP. I have fine tuned GPT-2 with a language model head on short jokes scrapped from reddit.

**Humor.ai tries to complete the sentences in a humourous way given some input words**

I tested my model on unseen words and sentences and got some really cool and surprising results

The first one is really hilarious given the fact that model doesn't know my name 😂😂 Language Model that can make you Laugh

![Image description](https://github.com/tanulsingh/Humour.ai/blob/master/Demo/sa.PNG)

![Image description](https://github.com/tanulsingh/Humour.ai/blob/master/Demo/Tanul.PNG)

![Image description](https://github.com/tanulsingh/Humour.ai/blob/master/Demo/saving.PNG)


# Data

The first challenge for any Machine Learning project is getting the data that would do the task. Fortunately I didn't have to do a  lot in getting the data , I found this awesome dataset on [Kaggle]( https://www.kaggle.com/abhinavmoudgil95/short-jokes) . It consists of short Jokes scrapped from reddit in well laid DataFrame

# Pre-Processing

Open GPT-2 is a transformer type architecture which uses the decoder of transformers . It is well known for it's language modelling tasks and thus I used it to create Humour.ai 

**There are two ways in which data can be presented to the model, depending on the objective you want to achieve**

* Joke generator 
* Humorous Sentence Completion

Let's look at these two seperately

### Joke Generation

In this task the model simply tries to generate jokes, given the length of joke and number of jokes you want it to generate.
Here we append 'JOKE:' at the start of every joke in our dataframe and '<|endoftext|>' at the end of each joke which tells our model that our joke has ended.
At the time of inference , we simply provide number of jokes and length of each joke and our model will print out jokes based on what it has learned

### Humorous Sentence Completion

This is something new , a simple tweak to above mentioned task . In this our model tries to complete a sentence in a humorous way given any input word or words it has never seen before.

For this task , I took only the Jokes in our dataset which were question,answer types and started with Why,When,How,etc. Then processed the data in this format<br><br>
<|soq|> question <|sep|> answer <|endoftext|> 

 It looks like an input to Question answering system , only the whole string is treated as one string , instead of getting different token_type_ids for Questions and Asnwers
 
 # Model
 
 I have used HuggingFace Library for GPT-2 Model and the whole code is written in Pytorch. I will be more than happy to share if someone takes this model and writes its equivalent in Keras/TF (that would be a good exercise) .The modelling and inference are easy to understand and self-explanatory if one reads the HuggingFace Docs.
 
 # HyperParameters
 
 I have tested two batch_sizes and two learning rates , the later works  better.It takes about 5 hours to train the first model for second task(Humorous Sentence Completion) on GPU's and  
 
| Task | Batch_Size | MAX_LEN | EPOCHS | Learning Rate| Train Time On GPU's | Train Time on TPU's|
|----------| ------------- | ------------- |------------- | ------------- | ---------|-----------|
|Humorous Sentence Completion|  32 | 64  | 4  | 3e-5  |4.5 hours| 2.5 hours|
|Humorous Sentence Completition| 16  | 64  | 4  | 2e-5  | 5.5 hours | 3 hours|
|Joke Generation | 32  | 64  | 4  | 3e-5  | 6.5 hours | 2.5 hours|
|Joke Generation | 16  | 64  | 4  | 2e-5  | 7.5 hours | 3 hours|

 
 # End Notes
 
* Feel Free to Fork, Experiment and play with the model . I have uploaded the code for the different tasks in different folders . 
 * **I will also be uploading trained weights so that anyone can load it and play with the model by just running the inference file**
 * I will be uploading the codde for taining on TPU's soon
 
