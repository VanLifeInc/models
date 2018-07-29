"""
Analyzer is the primary tool to analyze the results of our models.
"""

import operator
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from utils.load_images import load_images


class Analyzer:

    def __init__(self):
        pass

    def load_images(self,
                    image_types,
                    directory,
                    images_per_type,
                    image_size,
                    process,
                    model):
        """
        Load images and classes from a directory with the option to
        process the images and classes that will be used by a model.

        :param list[str] image_types: classes of images to load
        :param str directory: location of folders for each image class
        :param int images_per_type: number of images to load for each
            class
        :param int image_size: length and width of the loaded images
        :param bool process: option to process the images and classes so
            that they can be used to train and test the model
        :param keras.application model: model for which to process the
            images
        :return list[list]: loaded images
        :return list[str]: image classes
        """

        images, classes = load_images(image_types,
                                      directory,
                                      images_per_type,
                                      image_size,
                                      process,
                                      model)

        return images, classes

    def predict(self, model, images):
        """
        Predict the classes of a list of images using a model.

        :param keras.application model: will make the predictions
        :param list[list[int]] images: to have their classes predicted
        :return list[int]: the predicted class for the images
        """

        predictions = model.predict(images)

        return predictions

    def _get_index_for_max_value(self, values):
        """
        For a list of values, return the index of the largest value.

        :param list[int] values: to have the relevant indicies
            returned
        :return list[int]: the indicies related the largest values
        """

        indicies = [max(enumerate(values[i]), key=operator.itemgetter(1))[0] for i in range(len(values))]

        return indicies

    def _get_predicted_indicies(self, predictions, answers):
        """
        Get the indicies of the images that have been predicted
        correctly and incorrectly.

        :param list[int] predictions: predicted class of an image
        :param list[int] answers: true class of an image
        :return list[int] indicies_correct: correct predictions
        :return list[int] indicies_incorrect: incorrect predictions
        """

        indicies_predictions = self._get_index_for_max_value(predictions)
        indicies_actuals = self._get_index_for_max_value(answers)

        indicies_correct = []
        indicies_incorrect = []

        for i in range(len(indicies_predictions)):
            if indicies_predictions[i] == indicies_actuals[i]:
                indicies_correct.append(i)
            else:
                indicies_incorrect.append(i)

        return indicies_correct, indicies_incorrect

    def accuracy(self, predictions, answers, simple=True, image_types=None):
        """
        If simple, print the overall accuracy, otherwise, print the
        accuracy for each image type.

        :param list[int] predictions: predicted class of an image
        :param list[int] answers: true class of an image
        :param bool simple: detail of the accuracy to print
        :param list[int] image_types: which classes to have their
            accuracy printed
        """

        indicies_correct, indicies_incorrect = self._get_predicted_indicies(predictions, answers)

        if simple:
            accuracy = round(len(indicies_correct) / len(predictions) * 100, 2)
            print('Overall Accuracy: {}%'.format(accuracy))

        else:
            counts_correct = Counter(self._get_index_for_max_value(answers[indicies_correct]))
            counts_incorrect = Counter(self._get_index_for_max_value(answers[indicies_incorrect]))
            counts_total = counts_correct + counts_incorrect

            for i, image_type in enumerate(image_types):
                percent = round(counts_correct[i] / counts_total[i] * 100, 2)
                print('Accuracy per classification:')
                print('{}: {}/{}, {}%'.format(image_type,
                                              counts_correct[i],
                                              counts_total[i],
                                              percent))

    def show_results(self,
                     predictions,
                     answers,
                     image_types,
                     images,
                     correctness=None,
                     specific_image_types=None,
                     sample_count=None):
        """
        Print a specified group of images along with their predicted
        confidences and true classifications.

        :param list[int] predictions: predicted class of an image
        :param list[int] answers: true class of an image
        :param list[int] image_types: which classes to have their
            results analyzed
        :param list[list[int]] images: to have their results analyzed
        :param str correctness: whether to analyze only correctly or
            incorrectly predicted images
        :param list[str] specific_image_types: which classes of images
            to analyze
        :param int sample_count: number of images to analyze
        """

        if correctness != None:
            indicies_correct, indicies_incorrect = self._get_predicted_indicies(predictions, answers)
            indicies = indicies_correct if correctness == 'correct' else indicies_incorrect

        else:
            indicies = [i for i in range(len(predictions))]

        if specific_image_types != None:
            image_indicies = self._get_index_for_max_value(answers[indicies])
            indicies = [indicies[i] for i, index in enumerate(image_indicies)
                        if image_types[index] in specific_image_types]

        if sample_count != None:
            if len(indicies) < sample_count:
                print('NOTE!!!\nOnly {} images qualify for this filter.\n'.format(len(indicies)))
                sample_count = min(sample_count, len(indicies))

            indicies = np.random.choice(indicies, size=sample_count, replace=False)

        for i, index in enumerate(indicies):
            answer = image_types[self._get_index_for_max_value([answers[index]])[0]]
            prediction = image_types[self._get_index_for_max_value([predictions[index]])[0]]
            print('Image {}:'.format(i+1))
            print('Correct answer: {}'.format(answer))
            print('Prediction:\t{}\n'.format(prediction))
            for ii, value in enumerate(predictions[index]):
                print(image_types[ii], value)
            plt.figure()
            plt.imshow(np.uint8(images[index]))
            plt.show()
            print('------------------------------')
