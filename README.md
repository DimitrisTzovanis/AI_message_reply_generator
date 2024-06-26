# Artificial Intelligence that replicates conversations

## Members: Dimitris Tzovanis

#### RNN Neural Network that learns from text messages

This AI can generate text speech in a specific manner, by learning from messages that it has been fed.
To do this, I downloaded my Facebook Messenger data and located dozens of files with thousands of text messages.
I decrypted the messages and for every message I sent I gathered all the replies I recieved and exported the data to a txt.

Afterwards I fed the messages to a 2-layered-gru RNN that generates a number of characters for each input

To run this edit the Create_Dataset script and type the name of the folder with the json conversartion files

#### It features 2 gru layers for better text generation

It collects every 𝑆out(𝑙)𝑡 for each layer 𝑙 separately, and pass them to their respective layers as 𝑆in(𝑙)𝑡′ on the next iteration and in the same states variable, after the discrete co-ordinate value 𝑡′←𝑡 is advanced by the framework, to ensure that the invariant (𝑙) is conserved.


### Files

- Create_Dataset: finds all json files from a specific dierectory, decodes encoded facebook messenger data finds each message from a specific person, and lists all the replies from the other person
- rnn: creates the rnn model
- generate_response: uses the trained model to generate a quick response


### Parameters

- temperature: low = less randomness
- seq_length: number of characters each training example should have (40 is a typical messange exchange)
- BATCH_SIZE: number of batches (more can cause overfitting)
- embedding_dim: number of vectors for each word ( higher dimension might capture more information but at the cost of increased computational resources and possibly longer training times.)
- EPOCHS: training rounds (30 is enough)

