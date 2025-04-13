




This is a revisitation of my previous project, `LanguageTranslationModel` which I had worked on almost a year ago. 
THe purpose of this project is to revisit the basic to further stabalize my understandings, as I firmly believe building a strong foundational udnerstanding is crucial in this field. 
I reimplemented jsut about everything from scratched and added various adjustments. 

The most notable changes I made were: 
1. Sourced a larger and more accurate dataset
2. Improved implementation (Now that I revisit it, some of my original implementations were quite questionable. Among which modularity was a large problem and was one of the targets I focused on here)
3. Trained for a larger number of epoch with varied learning rate



Datasets are gathered from the following sources on HuggingFace:

load_dataset("swaption2009/20k-en-zh-translation-pinyin-hsk")
load_dataset("xzuyn/manythings-translations-alpaca")
load_dataset("zetavg/coct-en-zh-tw-translations-twp-300k")
load_dataset("Helsinki-NLP/opus-100", "en-zh")


Where it resulted in over a million en-zh pairs. 

I then merged my old en-zh dataset from my previous project, which was from
load_dataset("wmt16", "de-en")
and that added another 500k samples. 

In total, there was ~2.5M pairs. 

I then went through rigorous filtering, including but not limited to: 
English word frequency Filter- Set to 5, which trimmed ~400k examples
Min/max length filter after viewing stats
remove complete duplicate pairs
Repetitive words (Repeats more than 3 times in a row, few very, just a couple hundred)
Removing ones with many special chars. LIke *&^%$ etc.,
This successfully removes quite a lot of noisy data samples from the englihs side. The following are samples removed: 
---------------------
We don\\\\\\\'t have so many people from other places and we don\\\\\\\'t have Asian restaurants. The waiters in restaurants have uniforms.
The changes that have occurred in the U.S. housing market in the last decade aren\\\\\\\\\\\\\\\'t much different than the changes that occurred in Japan\\\\\\\\\\\\\\\'s boom market.
View <cake:nocache></cake:nocache> markup
pi * ti * 0..
The /*-{ and }-*/ delimiters for the method implementation
The argument of the function should be @@@ARGS@@@.
I am the living example of the \\\"triangle of life\\\".
sed -n 's/.*.
iptables [-t table] command [match] [target]
1/4 cup chopped pecans almonds or peanuts< o<="" span="">
---------------------

Where one can see that some are just plain noisy, while others seems to be commands/languages of some sort


At the end, I was left with just a little north of 1M sentence pairs to work with.


After the filtering stage I started training the model.
The model eventually pleataeud off at around a loss of 0.95 and 0.90 for Train and Validation loss respectively, which is a bit higher than before. 
Was expected it to be lower than before. 

During evaluation, I noticed that the meteor score was ~0.53, lower than the score of the old model which was at ~0.61
Quite a bit lower than I expected. 
After careful scrutiny, I noticeed that the dataset itself was at fault. 

Although filtering took care of a lot of low quality/bad samples, there was still quite a lot left. 
The problem was the that translation was highly inaccurate for a small chunk of the dataset, which affected the training/evaluation. 

That's when I remembered how I sourced the dataset in my original project. I gathered all the EN sentences and used a model to generate translation, ensuring high quality. That was the reason why I didn't directly find and use available en-zh, as they contain highly inaccurate translations. 
I ended up doing the same thing, and created the translation using a model (This time using gemini 2.0 flash to create the zh translation) 


I then ran the training again and got a much better result. 
Final train and val losses was 0.41 and 0.36 respsectively. A drastic imporovement. 


Below is the plot based on the saved output log file of the training losses: 

<p align="center">
  <img src="loss_plot.png" width="768">
</p>

So it's trained over 10 epochs, and one might notice the sudden dip in loss at epoch 8 (technically speaking, epoch 9)
That's because I trained the model using the learning rate described by the authors of Attention Is All You Need paper for the 
first 8 epochs, and used linspace(1e-4, 1e-5) for steps in epoch 9 and 10. This allows the model to escape the local minima and give a boost to the drop in loss


After evaluating on this model, the final metero score came out to be ~0.71

This is better, however still a bit lower than what I was expecting. 
After printing out a few samples that produced low scores, I found the following 3 problems: 


1. Numbers
The score is calculated based on a formulaic difference between the hypothesis (models' prediction) and reference (target translation)
I noticed some examples where it was
Source:     He gave you 5 billion dollars?
Reference:  他给了你50亿美元？
Hypothesis: 他给了你五十亿美元？

Source: Paragraph 16
Reference: 第16段
Hypothesis: 第十六段
Score: 0.17241379310344826

Both is correct, only difference is due to "16" vs "sixteen", which meteor docks off points for that. Quite heavily too, if the sentence is short, which is often the case.

2. Synonyms
The score wouldn't be able to account for synonyms. Though it's based on the `WordNet` model, that is designed for English words where it does account for synonyms. 
However the object here is Chinese, and so it would not be able to account for that. In this case, I highly suspect it's exact word matching, which further brings down the score for that. 
One of the example that I saw was:

Source: Explanation:
Reference: 解释：
Hypothesis: 说明：
Score: 0.25

To which I assume the only reason why it wasn't a 0 was probably due to the colon lol. Without it, that would very likely be 0.0 even though both are completely valid translations.


3. low-quality English Sentences
If the En sentence itself was bad, then so woudl the translation. 
Filtering did remove a lot of it, but it wasn't able to remove one's that sounded weird. I asked Gemini 2.0 Flash to evlaute it did remove some of those, however quite a fair amount still remained, even though I asked for it to be strict. 
For example, these en sentence were apparently good enough for the mmodel: 
- Congratulations to you it
- Finally could not help you borrow that book really sorry.
- ever i didnt know you had a baby, is it alive?
- I know you are not long for her not interested
- No matter how the bowl, as long as the protection of properly, will be in the years flow, snapped with true colors.

among others. Can one can see, some are weird and others are just plain incoherent. 

(All the above were from the dataset)


Anyways, those are the three primary problem causing the meteor score to be so 'low'. 
Though 0.71 is good enough, it's not an accurate reflection of the model's capability. 


