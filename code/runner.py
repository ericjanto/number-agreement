# coding: utf-8
import csv
import itertools
import os
import numpy as np
import sys
import time

from utils import *
from rnnmath import *
from sys import stdout
from model import Model
from rnn import RNN
from gru import GRU

import csv

class Runner(object):
    """
    This class implements the training loop for a Model (either an RNN or a GRU).
    Parameters such as hidden_size can be accessed via the model.

    You should implement code in the following functions:
        compute_loss 		->	compute the (cross entropy) loss between the desired output and predicted output for a given input sequence
        compute_loss_np     ->  compute the loss between the desired output and predicted output for a given input sequence for the number prediction task
        compute_mean_loss	->	compute the average loss over all sequences in a corpus
        compute_acc_np      ->  compute the

    Do NOT modify any other methods!
    Do NOT change any method signatures!
    """

    def __init__(self, model: Model):
        self.model = model

    def compute_loss(self, x, d):
        """
        compute the loss between predictions y for x, and desired output d.

        first predicts the output for x using the Model, then computes the loss w.r.t. d

        x		list of words, as indices, e.g.: [0, 4, 2]
        d		list of words, as indices, e.g.: [4, 2, 3]

        return loss		the combined loss for all words
        """

        loss = 0.0

        ##########################
        # --- your code here --- #
        ##########################

        # 0 -> di = onehot(0)
        # 4 -> xi = onehot(4)

        y = self.model.predict(x)[0]

        loss = -np.sum(np.log(y[np.arange(len(d)), d]))

        return loss

    def compute_loss_np(self, x, d):
        """
        compute the loss between predictions y for x, and desired output d.

        first predicts the output for x using the RNN, then computes the loss w.r.t. d

        x		list of words, as indices, e.g.: [0, 4, 2]
        d		a word, as indices, e.g.: [0]

        return loss		we only take the prediction from the last time step
        """

        loss = 0.0

        ##########################
        # --- your code here --- #
        ##########################

        y = self.model.predict(x)[0][-1]

        loss = -np.log(y[d[0]])

        return loss

    def compute_acc_np(self, x, d):
        """
        compute the accuracy prediction, y[t] compared to the desired output d.
        first predicts the output for x using the RNN, then computes the loss w.r.t. d

        x		list of words, as indices, e.g.: [0, 4, 2]
        d		a word class (plural/singular), as index, e.g.: [0] or [1]

        return 1 if argmax(y[t]) == d[0], 0 otherwise
        """

        ##########################
        # --- your code here --- #
        ##########################

        y = self.model.predict(x)[0][-1]

        return np.argmax(y) == d[0]

    def compute_mean_loss(self, X, D):
        """
        compute the mean loss between predictions for corpus X and desired outputs in corpus D.

        X		corpus of sentences x1, x2, x3, [...], each a list of words as indices.
        D		corpus of desired outputs d1, d2, d3 [...], each a list of words as indices.

        return mean_loss		average loss over all words in D
        """

        mean_loss = 0.0
        ##########################
        # --- your code here --- #
        ##########################
        loss_sum = sum([len(d) for d in D])
        mean_loss = (
            sum([self.compute_loss(X[i], D[i]) for i in range(len(X))]) / loss_sum
        )

        return mean_loss

    def train(
        self,
        X,
        D,
        X_dev,
        D_dev,
        epochs=10,
        learning_rate=0.5,
        anneal=5,
        back_steps=0,
        batch_size=100,
        min_change=0.0001,
        log=True,
    ):
        """
        train the model on some training set X, D while optimizing the loss on a dev set X_dev, D_dev

        DO NOT CHANGE THIS

        training stops after the first of the following is true:
            * number of epochs reached
            * minimum change observed for more than 2 consecutive epochs

        X				a list of input vectors, e.g., 		[[0, 4, 2], [1, 3, 0]]
        D				a list of desired outputs, e.g., 	[[4, 2, 3], [3, 0, 3]]
        X_dev			a list of input vectors, e.g., 		[[0, 4, 2], [1, 3, 0]]
        D_dev			a list of desired outputs, e.g., 	[[4, 2, 3], [3, 0, 3]]
        epochs			maximum number of epochs (iterations) over the training set. default 10
        learning_rate	initial learning rate for training. default 0.5
        anneal			positive integer. if > 0, lowers the learning rate in a harmonically after each epoch.
                        higher annealing rate means less change per epoch.
                        anneal=0 will not change the learning rate over time.
                        default 5
        back_steps		positive integer. number of timesteps for BPTT. if back_steps < 2, standard BP will be used. default 0
        batch_size		number of training instances to use before updating the RNN's weight matrices.
                        if set to 1, weights will be updated after each instance. if set to len(X), weights are only updated after each epoch.
                        default 100
        min_change		minimum change in loss between 2 epochs. if the change in loss is smaller than min_change, training stops regardless of
                        number of epochs left.
                        default 0.0001
        log				whether or not to print out log messages. (default log=True)
        """
        if log:
            stdout.write(
                "\nTraining model for {0} epochs\ntraining set: {1} sentences (batch size {2})".format(
                    epochs, len(X), batch_size
                )
            )
            stdout.write("\nOptimizing loss on {0} sentences".format(len(X_dev)))
            stdout.write(
                "\nVocab size: {0}\nHidden units: {1}".format(
                    self.model.vocab_size, self.model.hidden_dims
                )
            )
            stdout.write("\nSteps for back propagation: {0}".format(back_steps))
            stdout.write(
                "\nInitial learning rate set to {0}, annealing set to {1}".format(
                    learning_rate, anneal
                )
            )
            stdout.write("\n\ncalculating initial mean loss on dev set")
            stdout.flush()

        t_start = time.time()
        loss_function = self.compute_loss

        loss_sum = sum([len(d) for d in D_dev])
        initial_loss = (
            sum([loss_function(X_dev[i], D_dev[i]) for i in range(len(X_dev))])
            / loss_sum
        )

        if log or not log:
            stdout.write(": {0}\n".format(initial_loss))
            stdout.flush()

        prev_loss = initial_loss
        loss_watch_count = -1
        min_change_count = -1

        a0 = learning_rate

        best_loss = initial_loss
        self.model.save_params()
        best_epoch = 0

        for epoch in range(epochs):
            if anneal > 0:
                learning_rate = a0 / ((epoch + 0.0 + anneal) / anneal)
            else:
                learning_rate = a0

            if log:
                stdout.write(
                    "\nepoch %d, learning rate %.04f" % (epoch + 1, learning_rate)
                )
                stdout.flush()

            t0 = time.time()
            count = 0

            # use random sequence of instances in the training set (tries to avoid local maxima when training on batches)
            permutation = np.random.permutation(range(len(X)))
            if log:
                stdout.write("\tinstance 1")
            for i in range(len(X)):
                c = i + 1
                if log:
                    stdout.write("\b" * len(str(i)))
                    stdout.write("{0}".format(c))
                    stdout.flush()
                p = permutation[i]
                x_p = X[p]
                d_p = D[p]

                y_p, s_p = self.model.predict(x_p)
                if back_steps == 0:
                    self.model.acc_deltas(x_p, d_p, y_p, s_p)
                else:
                    self.model.acc_deltas_bptt(x_p, d_p, y_p, s_p, back_steps)

                if i % batch_size == 0:
                    self.model.scale_gradients_for_batch(batch_size)
                    self.model.apply_deltas(learning_rate)

            if len(X) % batch_size > 0:
                mod = len(X) % batch_size
                self.model.scale_gradients_for_batch(mod)
                self.model.apply_deltas(learning_rate)

            loss = (
                sum([loss_function(X_dev[i], D_dev[i]) for i in range(len(X_dev))])
                / loss_sum
            )

            if log:
                stdout.write("\tepoch done in %.02f seconds" % (time.time() - t0))
                stdout.write("\tnew loss: {0}".format(loss))
                stdout.flush()

            if loss < best_loss:
                best_loss = loss
                self.model.save_params()
                best_epoch = epoch

            # make sure we change the RNN enough
            if abs(prev_loss - loss) < min_change:
                min_change_count += 1
            else:
                min_change_count = 0
            if min_change_count > 2:
                print(
                    "\n\ntraining finished after {0} epochs due to minimal change in loss".format(
                        epoch + 1
                    )
                )
                break

            prev_loss = loss

        t = time.time() - t_start

        if min_change_count <= 2:
            print(
                "\n\ntraining finished after reaching maximum of {0} epochs".format(
                    epochs
                )
            )
        print(
            "best observed loss was {0}, at epoch {1}".format(
                best_loss, (best_epoch + 1)
            )
        )

        print("setting parameters to matrices from best epoch")
        self.model.set_best_params()

        return best_loss

    def train_np(
        self,
        X,
        D,
        X_dev,
        D_dev,
        epochs=10,
        learning_rate=0.5,
        anneal=5,
        back_steps=0,
        batch_size=100,
        min_change=0.0001,
        log=True,
        save_loss_acc_file = False
    ):
        """
        train the model on some training set X, D while optimizing the loss on a dev set X_dev, D_dev

        DO NOT CHANGE THIS

        training stops after the first of the following is true:
            * number of epochs reached
            * minimum change observed for more than 2 consecutive epochs

        X				a list of input vectors, e.g., 		[[5, 4, 2], [7, 3, 8]]
        D				a list of desired outputs, e.g., 	[[0], [1]]
        X_dev			a list of input vectors, e.g., 		[[5, 4, 2], [7, 3, 8]]
        D_dev			a list of desired outputs, e.g., 	[[0], [1]]
        epochs			maximum number of epochs (iterations) over the training set. default 10
        learning_rate	initial learning rate for training. default 0.5
        anneal			positive integer. if > 0, lowers the learning rate in a harmonically after each epoch.
                        higher annealing rate means less change per epoch.
                        anneal=0 will not change the learning rate over time.
                        default 5
        back_steps		positive integer. number of timesteps for BPTT. if back_steps < 2, standard BP will be used. default 0
        batch_size		number of training instances to use before updating the RNN's weight matrices.
                        if set to 1, weights will be updated after each instance. if set to len(X), weights are only updated after each epoch.
                        default 100
        min_change		minimum change in loss between 2 epochs. if the change in loss is smaller than min_change, training stops regardless of
                        number of epochs left.
                        default 0.0001
        log				whether or not to print out log messages. (default log=True)
        """
        if log:
            stdout.write(
                "\nTraining model for {0} epochs\ntraining set: {1} sentences (batch size {2})".format(
                    epochs, len(X), batch_size
                )
            )
            stdout.write("\nOptimizing loss on {0} sentences".format(len(X_dev)))
            stdout.write(
                "\nVocab size: {0}\nHidden units: {1}".format(
                    self.model.vocab_size, self.model.hidden_dims
                )
            )
            stdout.write("\nSteps for back propagation: {0}".format(back_steps))
            stdout.write(
                "\nInitial learning rate set to {0}, annealing set to {1}".format(
                    learning_rate, anneal
                )
            )
            stdout.flush()

        t_start = time.time()
        loss_function = self.compute_loss_np

        loss_sum = len(D_dev)
        initial_loss = (
            sum([loss_function(X_dev[i], D_dev[i]) for i in range(len(X_dev))])
            / loss_sum
        )
        initial_acc = sum(
            [self.compute_acc_np(X_dev[i], D_dev[i]) for i in range(len(X_dev))]
        ) / len(X_dev)

        if log or not log:
            stdout.write("\n\ncalculating initial mean loss on dev set")
            stdout.write(": {0}\n".format(initial_loss))
            stdout.write("calculating initial acc on dev set")
            stdout.write(": {0}\n".format(initial_acc))
            stdout.flush()

        prev_loss = initial_loss
        loss_watch_count = -1
        min_change_count = -1

        a0 = learning_rate

        best_loss = initial_loss
        self.model.save_params()
        best_epoch = 0

        for epoch in range(epochs):
            if anneal > 0:
                learning_rate = a0 / ((epoch + 0.0 + anneal) / anneal)
            else:
                learning_rate = a0

            if log:
                stdout.write(
                    "\nepoch %d, learning rate %.04f" % (epoch + 1, learning_rate)
                )
                stdout.flush()

            t0 = time.time()
            count = 0

            # use random sequence of instances in the training set (tries to avoid local maxima when training on batches)
            permutation = np.random.permutation(range(len(X)))
            if log:
                stdout.write("\tinstance 1")
            for i in range(len(X)):
                c = i + 1
                if log:
                    stdout.write("\b" * len(str(i)))
                    stdout.write("{0}".format(c))
                    stdout.flush()
                p = permutation[i]
                x_p = X[p]
                d_p = D[p]

                y_p, s_p = self.model.predict(x_p)
                if back_steps == 0:
                    self.model.acc_deltas_np(x_p, d_p, y_p, s_p)
                else:
                    self.model.acc_deltas_bptt_np(x_p, d_p, y_p, s_p, back_steps)

                if i % batch_size == 0:
                    self.model.scale_gradients_for_batch(batch_size)
                    self.model.apply_deltas(learning_rate)

            if len(X) % batch_size > 0:
                mod = len(X) % batch_size
                self.model.scale_gradients_for_batch(mod)
                self.model.apply_deltas(learning_rate)

            loss = (
                sum([loss_function(X_dev[i], D_dev[i]) for i in range(len(X_dev))])
                / loss_sum
            )
            acc = sum(
                [self.compute_acc_np(X_dev[i], D_dev[i]) for i in range(len(X_dev))]
            ) / len(X_dev)

            if log:
                stdout.write("\tepoch done in %.02f seconds" % (time.time() - t0))
                stdout.write("\tnew loss: {0}".format(loss))
                stdout.write("\tnew acc: {0}".format(acc))
                stdout.flush()

            if loss < best_loss:
                best_loss = loss
                best_acc = acc
                self.model.save_params()
                best_epoch = epoch

            if save_loss_acc_file:
                acc_loss.append([epoch, loss, acc])

            # make sure we change the RNN enough
            if abs(prev_loss - loss) < min_change:
                min_change_count += 1
            else:
                min_change_count = 0
            if min_change_count > 2:
                print(
                    "\n\ntraining finished after {0} epochs due to minimal change in loss".format(
                        epoch + 1
                    )
                )
                break

            prev_loss = loss

        t = time.time() - t_start

        if min_change_count <= 2:
            print(
                "\n\ntraining finished after reaching maximum of {0} epochs".format(
                    epochs
                )
            )
        print(
            "best observed loss was {0}, acc {1}, at epoch {2}".format(
                best_loss, best_acc, (best_epoch + 1)
            )
        )

        print("setting U, V, W to matrices from best epoch")
        self.model.set_best_params()

        return best_loss


if __name__ == "__main__":
    mode = sys.argv[1].lower()
    data_folder = sys.argv[2]
    np.random.seed(2018)

    if mode == "train-lm-rnn-parameter-search":
        """
        code for training language model.
        change this to different values, or use it to get you started with your own testing class
        """
        train_size = 1000
        dev_size = 1000
        vocab_size = 2000
        epochs = 10
        log = True
        batch_size = 100
        min_change = 0.0001

        # get the data set vocabulary
        vocab = pd.read_table(
            data_folder + "/vocab.wiki.txt",
            header=None,
            sep="\s+",
            index_col=0,
            names=["count", "freq"],
        )
        num_to_word = dict(enumerate(vocab.index[:vocab_size]))
        word_to_num = invert_dict(num_to_word)

        # calculate loss vocabulary words due to vocab_size
        fraction_lost = fraq_loss(vocab, word_to_num, vocab_size)
        print(
            "Retained %d words from %d (%.02f%% of all tokens)\n"
            % (vocab_size, len(vocab), 100 * (1 - fraction_lost))
        )

        docs = load_lm_dataset(data_folder + "/wiki-train.txt")
        S_train = docs_to_indices(docs, word_to_num, 1, 1)
        X_train, D_train = seqs_to_lmXY(S_train)

        # Load the dev set (for tuning hyperparameters)
        docs = load_lm_dataset(data_folder + "/wiki-dev.txt")
        S_dev = docs_to_indices(docs, word_to_num, 1, 1)
        X_dev, D_dev = seqs_to_lmXY(S_dev)

        X_train = X_train[:train_size]
        D_train = D_train[:train_size]
        X_dev = X_dev[:dev_size]
        D_dev = D_dev[:dev_size]

        # q = best unigram frequency from omitted vocab
        # this is the best expected loss out of that set
        q = vocab.freq[vocab_size] / sum(vocab.freq[vocab_size:])

        ##########################
        # --- your code here --- #
        ##########################
        hdims = [25, 50]
        lrs = [0.05, 0.1, 0.5]
        back_steps = [0, 2, 5]

        with open("finetuning.csv", "w") as f:
            writer = csv.writer(f)
            f.write("hdim,lr,back_step,loss,loss_adjusted\n")

            for hdim, lr, back_step in itertools.product(hdims, lrs, back_steps):
                run_loss = -1
                adjusted_loss = -1
                rnn = RNN(vocab_size, hdim, vocab_size)
                runner = Runner(rnn)
                run_loss = runner.train(
                    X_train,
                    D_train,
                    X_dev,
                    D_dev,
                    epochs=epochs,
                    learning_rate=lr,
                    anneal=0,
                    back_steps=back_step,
                    batch_size=batch_size,
                    min_change=min_change,
                    log=log,
                )
                adjusted_loss = adjust_loss(run_loss, fraction_lost, q)
                writer.writerow([hdim, lr, back_step, run_loss, adjusted_loss])

    if mode == "train-lm-rnn":
        """
        code for training language model.
        change this to different values, or use it to get you started with your own testing class
        """
        # ---
        # Best hyperparameters:
        # 25,0.5,5 (hdim,lr,lookback)
        # ---
        train_size = 25000
        dev_size = 1000
        vocab_size = 2000
        epochs = 10
        log = True
        batch_size = 100
        min_change = 0.0001

        hdims = int(sys.argv[3])
        lookback = int(sys.argv[4])
        lr = float(sys.argv[5])

        # get the data set vocabulary
        vocab = pd.read_table(
            data_folder + "/vocab.wiki.txt",
            header=None,
            sep="\s+",
            index_col=0,
            names=["count", "freq"],
        )
        num_to_word = dict(enumerate(vocab.index[:vocab_size]))
        word_to_num = invert_dict(num_to_word)

        # calculate loss vocabulary words due to vocab_size
        fraction_lost = fraq_loss(vocab, word_to_num, vocab_size)
        print(
            "Retained %d words from %d (%.02f%% of all tokens)\n"
            % (vocab_size, len(vocab), 100 * (1 - fraction_lost))
        )

        docs = load_lm_dataset(data_folder + "/wiki-train.txt")
        S_train = docs_to_indices(docs, word_to_num, 1, 1)
        X_train, D_train = seqs_to_lmXY(S_train)

        # Load the dev set (for tuning hyperparameters)
        docs = load_lm_dataset(data_folder + "/wiki-dev.txt")
        S_dev = docs_to_indices(docs, word_to_num, 1, 1)
        X_dev, D_dev = seqs_to_lmXY(S_dev)

        X_train = X_train[:train_size]
        D_train = D_train[:train_size]
        X_dev = X_dev[:dev_size]
        D_dev = D_dev[:dev_size]

        # q = best unigram frequency from omitted vocab
        # this is the best expected loss out of that set
        q = vocab.freq[vocab_size] / sum(vocab.freq[vocab_size:])

        ##########################
        # --- your code here --- #
        ##########################

        run_loss = -1
        adjusted_loss = -1

        rnn = RNN(vocab_size, hdim, vocab_size)
        runner = Runner(rnn)
        run_loss = runner.train(
            X_train,
            D_train,
            X_dev,
            D_dev,
            epochs=epochs,
            learning_rate=lr,
            anneal=0,
            back_steps=lookback,
            batch_size=batch_size,
            min_change=min_change,
            log=log,
        )

        dir = "matrices"

        np.save(os.path.join(dir, "rnn.U.npy"), rnn.U)
        np.save(os.path.join(dir, "rnn.V.npy"), rnn.V)
        np.save(os.path.join(dir, "rnn.W.npy"), rnn.W)

        adjusted_loss = adjust_loss(run_loss, fraction_lost, q)

        print("Unadjusted: %.03f" % np.exp(run_loss))
        print("Adjusted for missing vocab: %.03f" % np.exp(adjusted_loss))

    if mode == "evaluate-lm-rnn":
        """
        code for evaluating the rnn for q2b.
        """
        train_size = 25000
        dev_size = 1000
        vocab_size = 2000
        epochs = 10
        log = True
        batch_size = 100
        min_change = 0.0001
        hdim = 25

        # get the data set vocabulary
        vocab = pd.read_table(
            data_folder + "/vocab.wiki.txt",
            header=None,
            sep="\s+",
            index_col=0,
            names=["count", "freq"],
        )
        num_to_word = dict(enumerate(vocab.index[:vocab_size]))
        word_to_num = invert_dict(num_to_word)

        # calculate loss vocabulary words due to vocab_size
        fraction_lost = fraq_loss(vocab, word_to_num, vocab_size)
        print(
            "Retained %d words from %d (%.02f%% of all tokens)\n"
            % (vocab_size, len(vocab), 100 * (1 - fraction_lost))
        )

        # q = best unigram frequency from omitted vocab
        # this is the best expected loss out of that set
        q = vocab.freq[vocab_size] / sum(vocab.freq[vocab_size:])

        rnn = RNN(vocab_size, hdim, vocab_size)
        runner = Runner(rnn)

        # Load parameters
        dir = "matrices"
        rnn.U = np.load(os.path.join(dir, "rnn.U.npy"))
        rnn.V = np.load(os.path.join(dir, "rnn.V.npy"))
        rnn.W = np.load(os.path.join(dir, "rnn.W.npy"))

        # Evaluate on test set
        docs = load_lm_dataset(data_folder + "/wiki-test.txt")
        S_test = docs_to_indices(docs, word_to_num, 1, 1)
        X_test, D_test = seqs_to_lmXY(S_test)

        test_loss = runner.compute_mean_loss(X_test, D_test)
        adjusted_test_loss = adjust_loss(test_loss, fraction_lost, q)

        print("Mean loss: %.03f" % test_loss)
        # Get perplexity from test_loss and adjusted_test_loss
        print("Unadjusted perplexity: %.03f" % np.exp(test_loss))
        print("Adjusted perplexity: %.03f" % np.exp(adjusted_test_loss))

    if mode == "train-np-rnn":
        """
        starter code for parameter estimation.
        change this to different values, or use it to get you started with your own testing class
        """
        train_size = 10000
        dev_size = 1000
        vocab_size = 2000
        epochs = 10
        log = True
        batch_size = 100
        min_change = 0.0001
        lookback = 0

        hdims = [10, 25, 50]
        lr = 0.5

        # get the data set vocabulary
        vocab = pd.read_table(
            data_folder + "/vocab.wiki.txt",
            header=None,
            sep="\s+",
            index_col=0,
            names=["count", "freq"],
        )
        num_to_word = dict(enumerate(vocab.index[:vocab_size]))
        word_to_num = invert_dict(num_to_word)

        # calculate loss vocabulary words due to vocab_size
        fraction_lost = fraq_loss(vocab, word_to_num, vocab_size)
        print(
            "Retained %d words from %d (%.02f%% of all tokens)\n"
            % (vocab_size, len(vocab), 100 * (1 - fraction_lost))
        )

        # load training data
        sents = load_np_dataset(data_folder + "/wiki-train.txt")
        S_train = docs_to_indices(sents, word_to_num, 0, 0)
        X_train, D_train = seqs_to_npXY(S_train)

        X_train = X_train[:train_size]
        Y_train = D_train[:train_size]

        # load development data
        sents = load_np_dataset(data_folder + "/wiki-dev.txt")
        S_dev = docs_to_indices(sents, word_to_num, 0, 0)
        X_dev, D_dev = seqs_to_npXY(S_dev)

        X_dev = X_dev[:dev_size]
        D_dev = D_dev[:dev_size]

        ##########################
        # --- your code here --- #
        ##########################

        for hdim in hdims:
            acc = 0.0

            rnn = RNN(vocab_size, hdim, 2)
            runner = Runner(rnn)
            rnn_loss = runner.train_np(
                X_train,
                Y_train,
                X_dev,
                D_dev,
                epochs=epochs,
                learning_rate=lr,
                anneal=0,
                back_steps=lookback,
                batch_size=batch_size,
                min_change=min_change,
                log=log,
            )

            dir = "matrices"

            np.save(os.path.join(dir, f"rnn_np_hdim{hdim}.U.npy"), rnn.U)
            np.save(os.path.join(dir, f"rnn_np_hdim{hdim}.V.npy"), rnn.V)
            np.save(os.path.join(dir, f"rnn_np_hdim{hdim}.W.npy"), rnn.W)

            acc = sum(
                [runner.compute_acc_np(X_dev[i], D_dev[i]) for i in range(len(X_dev))]
            ) / len(X_dev)

            print(f"Accuracy {hdim}: %.03f" % acc)

    if mode == "train-np-gru":
        """
        starter code for parameter estimation.
        change this to different values, or use it to get you started with your own testing class
        """
        train_size = 10000
        dev_size = 1000
        vocab_size = 2000
        epochs = 10
        log = True
        batch_size = 100
        min_change = 0.0001
        lookback = 0

        hdims = [10, 25, 50]
        lr = 0.5

        # get the data set vocabulary
        vocab = pd.read_table(
            data_folder + "/vocab.wiki.txt",
            header=None,
            sep="\s+",
            index_col=0,
            names=["count", "freq"],
        )
        num_to_word = dict(enumerate(vocab.index[:vocab_size]))
        word_to_num = invert_dict(num_to_word)

        # calculate loss vocabulary words due to vocab_size
        fraction_lost = fraq_loss(vocab, word_to_num, vocab_size)
        print(
            "Retained %d words from %d (%.02f%% of all tokens)\n"
            % (vocab_size, len(vocab), 100 * (1 - fraction_lost))
        )

        # load training data
        sents = load_np_dataset(data_folder + "/wiki-train.txt")
        S_train = docs_to_indices(sents, word_to_num, 0, 0)
        X_train, D_train = seqs_to_npXY(S_train)

        X_train = X_train[:train_size]
        Y_train = D_train[:train_size]

        # load development data
        sents = load_np_dataset(data_folder + "/wiki-dev.txt")
        S_dev = docs_to_indices(sents, word_to_num, 0, 0)
        X_dev, D_dev = seqs_to_npXY(S_dev)

        X_dev = X_dev[:dev_size]
        D_dev = D_dev[:dev_size]

        ##########################
        # --- your code here --- #
        ##########################

        for hdim in hdims:
            acc = 0.0

            gru = GRU(vocab_size, hdim, 2)
            runner = Runner(gru)
            gru_loss = runner.train_np(
                X_train,
                Y_train,
                X_dev,
                D_dev,
                epochs=epochs,
                learning_rate=lr,
                anneal=0,
                back_steps=lookback,
                batch_size=batch_size,
                min_change=min_change,
                log=log,
            )

            dir = "matrices"

            np.save(os.path.join(dir, f"gru_np_hdim{hdim}.Ur.npy"), gru.Ur)
            np.save(os.path.join(dir, f"gru_np_hdim{hdim}.Vr.npy"), gru.Vr)
            np.save(os.path.join(dir, f"gru_np_hdim{hdim}.Uz.npy"), gru.Uz)
            np.save(os.path.join(dir, f"gru_np_hdim{hdim}.Vz.npy"), gru.Vz)
            np.save(os.path.join(dir, f"gru_np_hdim{hdim}.Uh.npy"), gru.Uh)
            np.save(os.path.join(dir, f"gru_np_hdim{hdim}.Vh.npy"), gru.Vh)
            np.save(os.path.join(dir, f"gru_np_hdim{hdim}.W.npy"), gru.W)

            acc = sum(
                [runner.compute_acc_np(X_dev[i], D_dev[i]) for i in range(len(X_dev))]
            ) / len(X_dev)

            print(f"Accuracy {hdim}: %.03f" % acc)

    if mode == "train-question-4":
        """
        starter code for parameter estimation.
        change this to different values, or use it to get you started with your own testing class
        """
        train_size = 10000
        dev_size = 1000
        vocab_size = 2000
        epochs = 10
        log = True
        batch_size = 100
        min_change = 0.0001
        lookbacks = [1, 3, 5, 10, 20, 30]

        hdim = 50
        lr = 0.5

        # get the data set vocabulary
        vocab = pd.read_table(
            data_folder + "/vocab.wiki.txt",
            header=None,
            sep="\s+",
            index_col=0,
            names=["count", "freq"],
        )
        num_to_word = dict(enumerate(vocab.index[:vocab_size]))
        word_to_num = invert_dict(num_to_word)

        # calculate loss vocabulary words due to vocab_size
        fraction_lost = fraq_loss(vocab, word_to_num, vocab_size)
        print(
            "Retained %d words from %d (%.02f%% of all tokens)\n"
            % (vocab_size, len(vocab), 100 * (1 - fraction_lost))
        )

        # load training data
        sents = load_np_dataset(data_folder + "/wiki-train.txt")
        S_train = docs_to_indices(sents, word_to_num, 0, 0)
        X_train, D_train = seqs_to_npXY(S_train)

        X_train = X_train[:train_size]
        Y_train = D_train[:train_size]

        # load development data
        sents = load_np_dataset(data_folder + "/wiki-dev.txt")
        S_dev = docs_to_indices(sents, word_to_num, 0, 0)
        X_dev, D_dev = seqs_to_npXY(S_dev)

        X_dev = X_dev[:dev_size]
        D_dev = D_dev[:dev_size]

        ##########################
        # --- your code here --- #
        ##########################
        print("Now training GRU")
        for lookback in lookbacks:
            acc = 0.0

            gru = GRU(vocab_size, hdim, 2)
            runner = Runner(gru)
            gru_loss = runner.train_np(
                X_train,
                Y_train,
                X_dev,
                D_dev,
                epochs=epochs,
                learning_rate=lr,
                anneal=0,
                back_steps=lookback,
                batch_size=batch_size,
                min_change=min_change,
                log=log,
            )

            dir = "matrices/question4/gru"

            np.save(os.path.join(dir, f"gru_np_lb_{lookback}.Ur.npy"), gru.Ur)
            np.save(os.path.join(dir, f"gru_np_lb_{lookback}.Vr.npy"), gru.Vr)
            np.save(os.path.join(dir, f"gru_np_lb_{lookback}.Uz.npy"), gru.Uz)
            np.save(os.path.join(dir, f"gru_np_lb_{lookback}.Vz.npy"), gru.Vz)
            np.save(os.path.join(dir, f"gru_np_lb_{lookback}.Uh.npy"), gru.Uh)
            np.save(os.path.join(dir, f"gru_np_lb_{lookback}.Vh.npy"), gru.Vh)
            np.save(os.path.join(dir, f"gru_np_lb_{lookback}.W.npy"), gru.W)

            acc = sum(
                [runner.compute_acc_np(X_dev[i], D_dev[i]) for i in range(len(X_dev))]
            ) / len(X_dev)

            print(f"Accuracy {hdim}: %.03f" % acc)

            losses = np.array(
                [runner.compute_loss_np(X_dev[i], D_dev[i]) for i in range(len(X_dev))]
            )
            losses_sorted = np.argsort(losses)

            accuracy = np.array(
                [runner.compute_acc_np(X_dev[i], D_dev[i]) for i in range(len(X_dev))]
            )
            accuracy_sorted = np.argsort(accuracy)

            if not os.path.exists("loss-sentences/gru/lookback-" + str(lookback)):
                os.makedirs("loss-sentences/gru/lookback-" + str(lookback))

            with open(
                "loss-sentences/gru/lookback-"
                + str(lookback)
                + "/sentences_with_lowest_losses.csv",
                "w",
            ) as f:
                header = ["sentence", "loss"]
                writer = csv.writer(f)
                writer.writerow(header)
                for i in range(10):
                    writer.writerow([X_dev[losses_sorted[i]], losses[losses_sorted[i]]])

            with open(
                "loss-sentences/gru/lookback-"
                + str(lookback)
                + "/sentences_with_highest_losses.csv",
                "w",
            ) as f:
                header = ["sentence", "loss"]
                writer = csv.writer(f)
                writer.writerow(header)
                for i in range(10):
                    writer.writerow(
                        [X_dev[losses_sorted[-i]], losses[losses_sorted[-i]]]
                    )

            with open(
                "loss-sentences/gru/lookback-"
                + str(lookback)
                + "/sentences_with_highest_accuracy.csv",
                "w",
            ) as f:
                header = ["sentence", "accuracy"]
                writer = csv.writer(f)
                writer.writerow(header)
                for i in range(10):
                    writer.writerow(
                        [X_dev[accuracy_sorted[-i]], accuracy[accuracy_sorted[-i]]]
                    )

            with open(
                "loss-sentences/gru/lookback-"
                + str(lookback)
                + "/sentences_with_lowest_accuracy.csv",
                "w",
            ) as f:
                header = ["sentence", "accuracy"]
                writer = csv.writer(f)
                writer.writerow(header)
                for i in range(10):
                    writer.writerow(
                        [X_dev[accuracy_sorted[i]], accuracy[accuracy_sorted[i]]]
                    )

        print("###########################################################")
        print("Now training RNN")

        for lookback in lookbacks:
            acc = 0.0

            rnn = RNN(vocab_size, hdim, 2)
            runner = Runner(rnn)
            rnn_loss = runner.train_np(
                X_train,
                Y_train,
                X_dev,
                D_dev,
                epochs=epochs,
                learning_rate=lr,
                anneal=0,
                back_steps=lookback,
                batch_size=batch_size,
                min_change=min_change,
                log=log,
            )

            dir = "matrices/question4/rnn"

            np.save(os.path.join(dir, f"rnn_np_lb_{lookback}.U.npy"), rnn.U)
            np.save(os.path.join(dir, f"rnn_np_lb_{lookback}.V.npy"), rnn.V)
            np.save(os.path.join(dir, f"rnn_np_lb_{lookback}.W.npy"), rnn.W)

            acc = sum(
                [runner.compute_acc_np(X_dev[i], D_dev[i]) for i in range(len(X_dev))]
            ) / len(X_dev)

            print(f"Accuracy {hdim}: %.03f" % acc)

            print()

            losses = np.array(
                [runner.compute_loss_np(X_dev[i], D_dev[i]) for i in range(len(X_dev))]
            )
            losses_sorted = np.argsort(losses)

            accuracy = np.array(
                [runner.compute_acc_np(X_dev[i], D_dev[i]) for i in range(len(X_dev))]
            )
            accuracy_sorted = np.argsort(accuracy)

            if not os.path.exists("loss-sentences/rnn/lookback-" + str(lookback)):
                os.makedirs("loss-sentences/rnn/lookback-" + str(lookback))

            with open(
                "loss-sentences/rnn/lookback-"
                + str(lookback)
                + "/sentences_with_lowest_losses.csv",
                "w",
            ) as f:
                header = ["sentence", "loss"]
                writer = csv.writer(f)
                writer.writerow(header)
                for i in range(10):
                    writer.writerow([X_dev[losses_sorted[i]], losses[losses_sorted[i]]])

            with open(
                "loss-sentences/rnn/lookback-"
                + str(lookback)
                + "/sentences_with_highest_losses.csv",
                "w",
            ) as f:
                header = ["sentence", "loss"]
                writer = csv.writer(f)
                writer.writerow(header)
                for i in range(10):
                    writer.writerow(
                        [X_dev[losses_sorted[-i]], losses[losses_sorted[-i]]]
                    )

            with open(
                "loss-sentences/rnn/lookback-"
                + str(lookback)
                + "/sentences_with_highest_accuracy.csv",
                "w",
            ) as f:
                header = ["sentence", "accuracy"]
                writer = csv.writer(f)
                writer.writerow(header)
                for i in range(10):
                    writer.writerow(
                        [X_dev[accuracy_sorted[-i]], accuracy[accuracy_sorted[-i]]]
                    )

            with open(
                "loss-sentences/rnn/lookback-"
                + str(lookback)
                + "/sentences_with_lowest_accuracy.csv",
                "w",
            ) as f:
                header = ["sentence", "accuracy"]
                writer = csv.writer(f)
                writer.writerow(header)
                for i in range(10):
                    writer.writerow(
                        [X_dev[accuracy_sorted[i]], accuracy[accuracy_sorted[i]]]
                    )

    if mode == "evaluate-q4":
        """
        code for evaluating the results of q4
        """
        train_size = 10000
        dev_size = 1000
        vocab_size = 2000
        epochs = 10
        log = True
        batch_size = 100
        min_change = 0.0001
        lookbacks = [1, 3, 5, 10, 20, 30]

        hdim = 50
        lr = 0.5

        vocab = pd.read_table(
            data_folder + "/vocab.wiki.txt",
            header=None,
            sep="\s+",
            index_col=0,
            names=["count", "freq"],
        )


        num_to_word = dict(enumerate(vocab.index[:vocab_size]))
        word_to_num = invert_dict(num_to_word)

        # calculate loss vocabulary words due to vocab_size
        fraction_lost = fraq_loss(vocab, word_to_num, vocab_size)
        print(
            "Retained %d words from %d (%.02f%% of all tokens)\n"
            % (vocab_size, len(vocab), 100 * (1 - fraction_lost))
        )

        # q = best unigram frequency from omitted vocab
        # this is the best expected loss out of that set
        q = vocab.freq[vocab_size] / sum(vocab.freq[vocab_size:])

        results_header = [
            "mean_loss_np",
            "accuracy",
            "model",
            "lookback",
        ]
        results = []

        rnn = RNN(vocab_size, hdim, 2)
        runner_rnn = Runner(rnn)

        gru = GRU(vocab_size, hdim, 2)
        runner_gru = Runner(gru)

        dir_rnn = "matrices/question4/rnn"
        dir_gru = "matrices/question4/gru"

        sents = load_np_dataset(data_folder + "/wiki-test.txt")
        S_dev = docs_to_indices(sents, word_to_num, 0, 0)
        X_dev, D_dev = seqs_to_npXY(S_dev)

        for lookback in [1, 3, 5, 10, 20, 30]:
            # RNN:
            # Load parameters
            rnn.U = np.load(os.path.join(dir_rnn, f"rnn_np_lb_{lookback}.U.npy"))
            rnn.V = np.load(os.path.join(dir_rnn, f"rnn_np_lb_{lookback}.V.npy"))
            rnn.W = np.load(os.path.join(dir_rnn, f"rnn_np_lb_{lookback}.W.npy"))


            # mean np loss
            mean_loss = sum([runner_rnn.compute_loss_np(X_dev[i], D_dev[i]) for i in range(len(X_dev))]) / len(X_dev)
            accuracy = sum([runner_rnn.compute_acc_np(X_dev[i], D_dev[i]) for i in range(len(X_dev))]) / len(X_dev)
            results.append(
                [
                    mean_loss,
                    accuracy,
                    "rnn",
                    lookback,
                ]
            )

            # GRU:
            # Load parameters
            gru.Ur = np.load(os.path.join(dir_gru, f"gru_np_lb_{lookback}.Ur.npy"))
            gru.Vr = np.load(os.path.join(dir_gru, f"gru_np_lb_{lookback}.Vr.npy"))
            gru.Uz = np.load(os.path.join(dir_gru, f"gru_np_lb_{lookback}.Uz.npy"))
            gru.Vz = np.load(os.path.join(dir_gru, f"gru_np_lb_{lookback}.Vz.npy"))
            gru.Uh = np.load(os.path.join(dir_gru, f"gru_np_lb_{lookback}.Uh.npy"))
            gru.Vh = np.load(os.path.join(dir_gru, f"gru_np_lb_{lookback}.Vh.npy"))
            gru.W = np.load(os.path.join(dir_gru, f"gru_np_lb_{lookback}.W.npy"))

            # Evaluate on test set
            mean_loss = sum([runner_gru.compute_loss_np(X_dev[i], D_dev[i]) for i in range(len(X_dev))]) / len(X_dev)
            accuracy = sum([runner_gru.compute_acc_np(X_dev[i], D_dev[i]) for i in range(len(X_dev))]) / len(X_dev)
            results.append(
                [
                    mean_loss,
                    accuracy,
                    "gru",
                    lookback,
                ]
            )

        with open("results-q4.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(results_header)
            writer.writerows(results)

    # get the data set vocabulary
    if mode == "get_sentences":

        vocab_size = 2000

        lookbacks = [1, 3, 5, 10, 20, 30]
        models = ["rnn", "gru"]
        fileNames = ["sentences_with_lowest_losses.csv",
                     "sentences_with_highest_losses.csv",
                     "sentences_with_highest_accuracy.csv",
                     "sentences_with_lowest_accuracy.csv"]

        vocab = pd.read_table(
            data_folder + "/vocab.wiki.txt",
            header=None,
            sep="\s+",
            index_col=0,
            names=["count", "freq"],
        )

        num_to_word = dict(enumerate(vocab.index[:vocab_size]))
        word_to_num = invert_dict(num_to_word)

        for model in models:
            print()
            print()
            print("##########################################################################################")
            print("##########################################################################################")
            print("SENTENCES FOR MODEL: " + model)

            for lookback in lookbacks:
                print()
                print("##########################################################################################")
                print("LOOKBACK: " + str(lookback))
                for fileName in fileNames:
                    print()
                    print(fileName)
                    file = open("loss-sentences/" + model +
                                "/lookback-" + str(lookback) +
                                "/" + fileName, "r")
                    csvreader = csv.reader(file)
                    _ = next(csvreader)

                    sentences = []
                    stats = []

                    for row in csvreader:
                        sentence = row[0].replace("[", "").replace("]", "")
                        sentence = sentence.split()
                        sentence = np.array(sentence, dtype=int)
                        sentence = [num_to_word[wordNum] for wordNum in sentence]
                        sentences.append(sentence)
                        stats.append(row[1])

                        print(sentence)
                        print(row[1])

    if mode == "convergence-test":
        train_size = 10000
        dev_size = 1000
        vocab_size = 2000
        epochs = 50
        log = True
        batch_size = 100
        min_change = 0.0001
        lookback = 10
        hdim = 50

        lr = 0.5

        # get the data set vocabulary
        vocab = pd.read_table(
            data_folder + "/vocab.wiki.txt",
            header=None,
            sep="\s+",
            index_col=0,
            names=["count", "freq"],
        )

        num_to_word = dict(enumerate(vocab.index[:vocab_size]))
        word_to_num = invert_dict(num_to_word)

        # calculate loss vocabulary words due to vocab_size
        fraction_lost = fraq_loss(vocab, word_to_num, vocab_size)
        print(
            "Retained %d words from %d (%.02f%% of all tokens)\n"
            % (vocab_size, len(vocab), 100 * (1 - fraction_lost))
        )

        # load training data
        sents = load_np_dataset(data_folder + "/wiki-train.txt")
        S_train = docs_to_indices(sents, word_to_num, 0, 0)
        X_train, D_train = seqs_to_npXY(S_train)

        X_train = X_train[:train_size]
        Y_train = D_train[:train_size]

        # load development data
        sents = load_np_dataset(data_folder + "/wiki-dev.txt")
        S_dev = docs_to_indices(sents, word_to_num, 0, 0)
        X_dev, D_dev = seqs_to_npXY(S_dev)

        X_dev = X_dev[:dev_size]
        D_dev = D_dev[:dev_size]

        rnn = RNN(vocab_size, hdim, 2)
        runner = Runner(rnn)

        rnn_loss = runner.train_np(
            X_train,
            Y_train,
            X_dev,
            D_dev,
            epochs=epochs,
            learning_rate=lr,
            anneal=0,
            back_steps=lookback,
            batch_size=batch_size,
            min_change=min_change,
            log=log,
        )

    if mode == "question-5":
        train_size = 10000
        dev_size = 1000
        vocab_size = 2000
        epochs = 10
        log = True
        batch_size = 100
        min_change = 0.0001
        lookback = 10

        hdim = 50
        lr = 0.5

        anneal_values = [0.1,0.5,1,2,4,8,16,32]

        # get the data set vocabulary
        vocab = pd.read_table(
            data_folder + "/vocab.wiki.txt",
            header=None,
            sep="\s+",
            index_col=0,
            names=["count", "freq"],
        )
        num_to_word = dict(enumerate(vocab.index[:vocab_size]))
        word_to_num = invert_dict(num_to_word)

        # calculate loss vocabulary words due to vocab_size
        fraction_lost = fraq_loss(vocab, word_to_num, vocab_size)
        print(
            "Retained %d words from %d (%.02f%% of all tokens)\n"
            % (vocab_size, len(vocab), 100 * (1 - fraction_lost))
        )

        # load training data
        sents = load_np_dataset(data_folder + "/wiki-train.txt")
        S_train = docs_to_indices(sents, word_to_num, 0, 0)
        X_train, D_train = seqs_to_npXY(S_train)

        X_train = X_train[:train_size]
        Y_train = D_train[:train_size]

        # load development data
        sents = load_np_dataset(data_folder + "/wiki-dev.txt")
        S_dev = docs_to_indices(sents, word_to_num, 0, 0)
        X_dev, D_dev = seqs_to_npXY(S_dev)

        X_dev = X_dev[:dev_size]
        D_dev = D_dev[:dev_size]

        global acc_loss

        for anneal in anneal_values:
            acc = 0.0
            acc_loss = []

            rnn = RNN(vocab_size, hdim, 2)
            runner = Runner(rnn)
            rnn_loss = runner.train_np(
                X_train,
                Y_train,
                X_dev,
                D_dev,
                epochs=epochs,
                learning_rate=lr,
                anneal=anneal,
                back_steps=lookback,
                batch_size=batch_size,
                min_change=min_change,
                log=log,
                save_loss_acc_file=True
            )

            with open(f"question_5_anneal_{anneal}_loss_acc.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "loss", "acc"])
                writer.writerows(acc_loss)

            dir = "matrices/question5"

            np.save(os.path.join(dir, f"rnn_np_anneal_{lookback}.U.npy"), rnn.U)
            np.save(os.path.join(dir, f"rnn_np_anneal_{lookback}.V.npy"), rnn.V)
            np.save(os.path.join(dir, f"rnn_np_anneal_{lookback}.W.npy"), rnn.W)

            acc = sum(
                [runner.compute_acc_np(X_dev[i], D_dev[i]) for i in range(len(X_dev))]
            ) / len(X_dev)

            print(f"Accuracy {hdim}: %.03f" % acc)
