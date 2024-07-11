# CNN/DailyMail & T5 Example

This directory contains scripts for fine-tuning T5 and computing influence scores on the CNN/DailyMail dataset. The pipeline is motivated from [this HuggingFace Example](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization).
To begin, install the necessary packages:

```bash
pip install -r requirements.txt
```

## Training

To fine-tune T5 on CNN/DailyMail, run the following command:

```bash
python train.py --checkpoint_dir ./checkpoints \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-05 \
    --weight_decay 0.01 \
    --num_train_epochs 3 \
    --seed 1004
```

This will fine-tune the model using the specified hyperparameters and save the final checkpoint in the `./checkpoints` directory.

## Computing Pairwise Influence Scores

To calculate pairwise influence scores on 10 query data points using `ekfac`, run:

```bash
python analyze.py --factor_batch_size 64 \
    --query_batch_size 10 \
    --train_batch_size 128 \
    --use_half_precision \
    --checkpoint_dir ./checkpoints \
    --factor_strategy ekfac
```

Alternative options for `factor_strategy` include `identity`, `diagonal`, and `kfac`. On an A100 (80GB), computing the pairwise scores (including EKFAC factors) takes approximately 1 hour:

```
----------------------------------------------------------------------------------------------------------------------------------
|  Action                       |  Mean duration (s)    |  Num calls            |  Total time (s)       |  Percentage %         |
----------------------------------------------------------------------------------------------------------------------------------
|  Total                        |  -                    |  11                   |  3397.1               |  100 %                |
----------------------------------------------------------------------------------------------------------------------------------
|  Compute Pairwise Score       |  1905.6               |  1                    |  1905.6               |  56.093               |
|  Fit Lambda                   |  747.5                |  1                    |  747.5                |  22.004               |
|  Fit Covariance               |  734.03               |  1                    |  734.03               |  21.607               |
|  Perform Eigendecomposition   |  8.4236               |  1                    |  8.4236               |  0.24796              |
|  Save Eigendecomposition      |  0.79164              |  1                    |  0.79164              |  0.023303             |
|  Save Covariance              |  0.60366              |  1                    |  0.60366              |  0.01777              |
|  Save Lambda                  |  0.1514               |  1                    |  0.1514               |  0.0044566            |
|  Load All Factors             |  0.027977             |  1                    |  0.027977             |  0.00082354           |
|  Save Pairwise Score          |  0.01082              |  1                    |  0.01082              |  0.00031851           |
|  Load Covariance              |  0.010015             |  1                    |  0.010015             |  0.0002948            |
|  Load Eigendecomposition      |  0.0096806            |  1                    |  0.0096806            |  0.00028497           |
----------------------------------------------------------------------------------------------------------------------------------
```

## Inspecting Top Influential Sequences

The `inspect_examples.py` script prints top influential sequences for a given query.

```
Query Data Example:
 Input: summarize: (CNN)My vote for Father of the Year goes to Curt Schilling. The former Major League Baseball pitcher recently fired off a series of fastballs and mowed down a group of Twitter trolls who made the mistake of tweeting vulgar and sexually-explicit comments about Schilling's teenage daughter. The drama started, innocently enough, on February 25, when Schilling played the role of a proud father. He sent a tweet congratulating his daughter, Gabby, on being accepted to Salve Regina University, where she'll play softball. It read: "Congrats to Gabby Schilling who will pitch for the Salve Regina Seahawks next year!! — Curt Schilling (@gehrig38)" Almost immediately, responses came in from young men, complete strangers who apparently followed Schilling on Twitter. The tweets quickly went from immature, to creepy, to repugnant. Threats of rape were common. The tweets were deleted, and the accounts were closed after this story went viral. But not before Schilling captured some of the images and posted them on his blog. What was said about 17-year-old Gabby Schilling wasn't just obnoxious. It was vile and obscene. What was said wasn't just mean and ugly. It was threatening and scary. As a parent, it's the kind of thing that makes you rethink your opposition to public caning as a logical punishment for such transgressions. These misogynistic cowards may have thought they could hide in the darkness of anonymity, the sort that many have come to expect from social media sites, where you feel free to be a despicable human being because, you think, no one will ever find out who you really are and hold you accountable for your words. If so, they thought wrong. They couldn't hide. They were found out, and they got the throttling they so richly deserved. Thanks to dad. According to Schilling, who made it his mission to track down these cretins and make sure those they associate with know who they really are, two people have already paid a price due to their tweets. One was a student disc jockey at a community college in New Jersey, who was suspended, and the other was a part-time ticket seller for the New York Yankees, who was fired. Concerned that this is an example of exactly the kind of cyberbullying that leads some teenagers to commit suicide, Schilling is also thinking about taking legal action against some of the other people involved. Bravo for him. I'm sure that, all across America, dads with daughters -- after reading some of the horrible things that were said about this young girl -- are marveling at Schilling's self-control. I have two daughters of my own, and he's a better man than me. If ever there was a case where profanity-spewing malcontents deserved to have their mouths washed out with soap, this is it. So what additional insights can we draw, and what larger lessons can we learn, from this unexpected but predictable collision of old-fashioned parenthood and newfangled media? There are a few. The first is about accountability, the very thing that the young men who posted these hurtful messages were trying to avoid. But Schilling wouldn't let them. At their best, social media sites like Twitter, Facebook, Instagram and others allow the sharing the information and the building of a sense of community. At their worst, they become digital sandboxes and locker rooms where people think have a license to misbehave without having to worry about consequences. We need to applaud efforts like this that promote greater online accountability. There's also something to be said about protective parents, and how essential they are to a working society. We should still be concerned about those overprotective parents who hover like helicopters from little league to job interviews. We shouldn't bubblewrap our kids, and keep them from playing outdoors, and then sit around wondering why they're soft, timid, and risk-averse. But protective parents -- the kind who shield their kids from real danger -- never go out of style. A parent's top job is to protect his children. Schilling did his job. Finally, it's worth reminding everyone that freedom of expression does not mean freedom from rules, standards, and expectations that should guide your behavior. There are things you don't say. There are boundaries, ways that we expect you to behave so you don't terrorize other people or bring shame upon yourself, your friends, and</s>
 Label: Ruben Navarrette: Schilling deserves praise for taking on online haters for offensive comments about his daughter. Navarrette: In protecting his child, Schilling set a model for parenting and taught us a lesson about social media.</s>

Top Influential Example:
 Input: summarize: (CNN) -- What is it with juries in high-profile cases in Southern California? Over the years, they've become a national joke. But no one is laughing. Instead, with each travesty of justice and every acquittal that should have been a conviction, you're left wondering just what trial these 12 folks were watching and questioning whether we should have higher standards for who sits on a jury. Sometimes, the juries in local and state courts get it wrong, and the Justice Department must step in and make it right. Think back to the acquittal in April 1992 of four Los Angeles police officers who, one year earlier, savagely beat motorist Rodney King. They walked out of a courtroom in Simi Valley, California, as free men -- sparking days of rioting, looting and violence. At the time, the conventional thinking on newspaper editorial pages and on talk radio was the jurors in that largely white suburb of Los Angeles, which was itself home to many active-duty and retired police officers, saw the police force as their line of defense against undesirables like King. So naturally, the argument went, they would cut them some slack. The officers were tried again, and convicted in federal court of violating King's civil rights. Justice was finally served. Here we go again. There hasn't been much civil unrest over what happened to Kelly Thomas, the homeless and mentally ill man who -- on July 5, 2011 -- was beaten to death by a swarm of police officers in Fullerton, California. But now that the verdict is in, literally, on the two former officers who were charged in his death, there is plenty of outrage on talk radio, online and in other public forums. Another 12 people who swore an oath to consider the evidence and the law and make sure that justice is served appear to have gotten it terribly wrong. This week, that jury in Santa Ana, California -- a city about 30 miles southeast of Los Angeles -- produced a wave of gasps in the courtroom when it announced that it had found Manuel Ramos, who had been charged with second-degree murder and involuntary manslaughter, and Jay Cicinelli, who was charged with involuntary manslaughter and excessive use of force, not guilty on all counts. What? The beating was caught on a surveillance tape. When you watch those 33 minutes of footage, assuming you can stomach the experience, it's hard to believe that anyone could declare the perpetrators "not guilty." The surveillance camera footage shows Thomas being beaten and stunned with a Taser by police until he was unrecognizable and unconscious. You see a defenseless and compliant young man screaming in pain, saying he's sorry and pleading for help from his father. His words will haunt you, "Daddy, help! They're killing me!" According to prosecutors, the young man suffered brain injuries, facial fractures, broken ribs and extensive bruises and abrasions. He wound up lying in a pool of blood. He died five days later. This was not a by-the-book case of police officers using all necessary force to subdue a suspect who was resisting arrest -- a suspect, by the way, who had committed no crime. This was not, as Ramos' attorney claimed, a case of police offices simply "doing their job" with "no malice in their heart." Check the video. Early on in the confrontation, Ramos appears to tell the young man who is sitting on the ground: "You see my fists? They're getting ready to f--- you up!" Another officer is heard telling a comrade: "We ran out of options so I got to the end of my Taser and I... smashed his face to hell." There is the malice. This was abuse of power and an instance of bullying behind a badge. It happens more than we'd like to think in America. But this time, it went too far. And a man died, and a family was shattered. Yet, the jury somehow missed all this? How does this happen? In Los Angeles, people are saying that the mentally ill are the new Rodney King. In the same way that the jury in Simi Valley was inclined to back the officers who it saw as protecting them from people like King, now the jury in Santa Ana is backing the officers who it counts on to prod people like Thomas to move along, leave the streets, and get out of sight. It's a plausible explanation</s>
 Label: Ruben Navarrette: Too many high-profile cases in California produce travesties of justice. He says jury acquitted two ex-cops in malicious beating death captured on video. He says case showed abuse of power, bullying behind badge; happens too often in U.S. Navarrette: Only one place left that can right this wrong: The Justice Department.</s>
```