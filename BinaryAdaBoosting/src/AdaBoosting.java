
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

class classifier {
	hypothesis h1;
	double alpha;
	double fractionalError;
	double boundOnEt;
}

class example {
	double x;
	int y;
	double prob;
	int num;
	int h; // classifier classification
	double f; // alpha * h value
}

class hypothesis {
	double threshold;
	boolean left;
	double weightedTrainingError;
}

public class AdaBoosting {

	static int noOfIter;
	static double epsilon;

	static ArrayList<example> readInput() throws FileNotFoundException {
		Scanner in = new Scanner(new File("adaboost-5.txt"));
		noOfIter = in.nextInt();
		int noOfEx = in.nextInt();
		epsilon = in.nextDouble();
		ArrayList<example> listOfExamples = new ArrayList<>();
		for (int i = 0; i < noOfEx; i++) {
			example ex = new example();
			ex.x = in.nextDouble();
			ex.num = i;
			listOfExamples.add(ex);
		}
		for (int i = 0; i < noOfEx; i++) {
			listOfExamples.get(i).y = in.nextInt();
		}
		for (int i = 0; i < noOfEx; i++) {
			listOfExamples.get(i).prob = in.nextDouble();
		}

		return listOfExamples;

	}

	static void printList(ArrayList<example> listOfExamples) {
		for (int i = 0; i < listOfExamples.size(); i++) {
			System.out.println(listOfExamples.get(i).num + " " + listOfExamples.get(i).x + " " + listOfExamples.get(i).y
					+ " " + listOfExamples.get(i).h + " " + listOfExamples.get(i).prob);
		}
	}

	static hypothesis getHypothesis(ArrayList<example> listOfExamples) {
		int noOfMisClassifiedEx = 0;
		double sumOfProbOfMisClassEx;
		double minOfAllThresholds = Double.MAX_VALUE;
		int minOfAllEx = 0;
		boolean globalLeft = false;
		for (int i = 0; i <= listOfExamples.size() - 1; i++) {
			// splitting b/w i-1 and i and counting the probabilities
			// for misclassified examples counting the probabilities
			// for examples below the threshold we need to check two
			// cases here one all the ex below the threshold are positive
			// and the ones above the threshold are negative
			// and the other case is the reverse . Among these two
			// choose the one with minimum no of errors
			boolean leftPos = true;
			boolean right;
			boolean left = right = false;
			int noOfCases = 2;
			double minProbSum = Double.MAX_VALUE;
			while (noOfCases != 0) {
				sumOfProbOfMisClassEx = 0;
				if (noOfCases != 2) {
					leftPos = false;
				}
				// in the left partition
				for (int j = 0; j < i; j++) {
					if (leftPos) {
						// if lefts are positive and the classification of the
						// ex
						// in the left fold is not 1 then it is misclassified
						if (listOfExamples.get(j).y != 1) {
							sumOfProbOfMisClassEx = sumOfProbOfMisClassEx + listOfExamples.get(j).prob;

						}
					} else if (!leftPos) {
						// if rights are positive and the classification of the
						// ex
						// in the left fold is not -1 then it is misclassified
						if (listOfExamples.get(j).y != -1) {
							sumOfProbOfMisClassEx = sumOfProbOfMisClassEx + listOfExamples.get(j).prob;

						}

					}

				}
				// counting probabilities for examples above the threshold
				for (int j = listOfExamples.size() - 1; j >= i; j--) {
					if (!leftPos) {
						// if rights are positive and the classification of the
						// ex
						// in the right fold is not 1 then it is misclassified
						if (listOfExamples.get(j).y != 1) {
							sumOfProbOfMisClassEx = sumOfProbOfMisClassEx + listOfExamples.get(j).prob;

						}
					} else if (leftPos) {
						// if lefts are positive and the classification of the
						// ex
						// in the right fold is not -1 then it is misclassified
						if (listOfExamples.get(j).y != -1) {
							sumOfProbOfMisClassEx = sumOfProbOfMisClassEx + listOfExamples.get(j).prob;

						}

					}

				}
				if (minProbSum > sumOfProbOfMisClassEx) {
					minProbSum = sumOfProbOfMisClassEx;
					if (leftPos) {
						left = true;
					} else {
						left = false;
					}

				}
				noOfCases--;
			}

			// System.out.println(minProbSum);
			if (minOfAllThresholds > minProbSum) {

				minOfAllThresholds = minProbSum;
				minOfAllEx = i;
				if (left) {
					globalLeft = true;
				} else {
					globalLeft = false;
				}
			}

		}
		hypothesis hCurrent = new hypothesis();
		if (minOfAllEx != 0) {
			hCurrent.threshold = (listOfExamples.get(minOfAllEx).x + listOfExamples.get(minOfAllEx - 1).x) / 2;
		} else {
			hCurrent.threshold = (listOfExamples.get(minOfAllEx).x - 0.5);
		}
		// System.out.println(hCurrent.threshold);
		hCurrent.left = globalLeft;
		hCurrent.weightedTrainingError = minOfAllThresholds;
		return hCurrent;
	}

	static/*
			 * function to print the boosted classifier
			 */
	void printBoostedClassifier(ArrayList<classifier> boostedClassifier) {

		for (int i = 0; i < boostedClassifier.size(); i++) {
			if (boostedClassifier.get(i).h1.left) {
				System.out.print(
						boostedClassifier.get(i).alpha + " * ( x < " + boostedClassifier.get(i).h1.threshold + " )");
			} else {
				System.out.print(
						boostedClassifier.get(i).alpha + " * ( x > " + boostedClassifier.get(i).h1.threshold + " )");

			}
			if (boostedClassifier.size() > 1 && i < boostedClassifier.size() - 1)
				System.out.print(" + ");

		}
		System.out.println();
	}

	public static void main(String[] args) throws FileNotFoundException {
		// TODO Auto-generated method stub
		ArrayList<example> listOfExamples;
		// reading input from the input file input.txt.txt
		listOfExamples = readInput();
		// print the list that is just read
		// printList(listOfExamples);
		// get the hypothesis from the examples
		hypothesis h1;
		// debug and figure out the sign properly
		double[] alphas = new double[noOfIter];
		double[] Zt = new double[noOfIter];
		// boosted classifier
		// this will hold the hypothesis and the corresponding goodness weights
		// for
		// each iteration
		ArrayList<classifier> boostedClassifier = new ArrayList<>();
		// wrongCount is used to count the wrongly classified examples by the
		// classifier
		double wrongCount;
		double Et;
		double bound;
		for (int i = 0; i < noOfIter; i++) {
			System.out.println("Iteration " + i);
			h1 = getHypothesis(listOfExamples);
			updateClassicationForHypotheis(listOfExamples, h1);

			/*
			 * if(h1.left) System.out.println("example is positive if x is < "
			 * +h1.threshold + " with error " + h1.weightedTrainingError); else
			 * System.out.println("example is positive if x is > " +h1.threshold
			 * + " with error " + h1.weightedTrainingError);
			 */
			/*
			 * if the classifier is not week we can break and run the algorithm
			 * again
			 */
			if (h1.weightedTrainingError == 0) {
				System.out.println("classifier is not week Run the algo again");
				break;
			} else if (h1.weightedTrainingError > 0.5) {
				System.out.println("weighted error is > 0.5 rerun the algo again");
				break;
			}

			// computing the goodness weight of hypothesis
			alphas[i] = (0.5) * (Math.log((1 - h1.weightedTrainingError) / h1.weightedTrainingError));

			// computing the Zt value

			Zt[i] = 2 * Math.sqrt(h1.weightedTrainingError * (1 - h1.weightedTrainingError));

			// computing the updated probabilities
			wrongCount = 0;
			for (int j = 0; j < listOfExamples.size(); j++) {
				// if(h1.left)
				listOfExamples.get(j).prob = (listOfExamples.get(j).prob
						* Math.pow(Math.E, -(alphas[i] * listOfExamples.get(j).y * listOfExamples.get(j).h))) / Zt[i];
				/*
				 * if (listOfExamples.get(j).y != listOfExamples.get(j).h) {
				 * wrongCount++;
				 */
				// }
			}
			// adding h1,alpha from each iteration to the boostedClassifier
			classifier localClassifier = new classifier();
			localClassifier.h1 = h1;
			localClassifier.alpha = alphas[i];
			double f[] = new double[listOfExamples.size()];
			for (int k = 0; k < listOfExamples.size(); k++) {
				if (h1.left) {
					if (listOfExamples.get(k).x < h1.threshold) {
						listOfExamples.get(k).f = listOfExamples.get(k).f + alphas[i];
					} else {
						listOfExamples.get(k).f = listOfExamples.get(k).f - alphas[i];
					}
				} else {
					if (listOfExamples.get(k).x > h1.threshold) {
						listOfExamples.get(k).f = listOfExamples.get(k).f + alphas[i];
					} else {
						listOfExamples.get(k).f = listOfExamples.get(k).f - alphas[i];
					}
				}

			}

			for (int k = 0; k < listOfExamples.size(); k++) {
				// System.out.println(listOfExamples.get(k).f);
				if (listOfExamples.get(k).f < 0 && listOfExamples.get(k).y != -1) {
					wrongCount++;
				} else if (listOfExamples.get(k).f > 0 && listOfExamples.get(k).y != 1) {
					wrongCount++;
				}
			}
			// System.out.println(wrongCount);
			boostedClassifier.add(localClassifier);
			// computing Et
			// computing fractional error of the boosted classifier Et
			localClassifier.fractionalError = wrongCount / listOfExamples.size();
			bound = 1;
			// computing the bound on Et

			for (int k = 0; k <= i; k++) {
				bound = bound * Zt[k];
			}
			localClassifier.boundOnEt = bound;
			System.out.print("The selected weak classifier: ");
			if (h1.left) {
				System.out.println(" x < " + h1.threshold);
			} else {
				System.out.println(" x > " + h1.threshold);
			}

			System.out.println("The error of Ht: " + h1.weightedTrainingError);
			System.out.println("The weight of Ht: " + alphas[i]);
			System.out.println("The probabilities normalization factor Zt: " + Zt[i]+" ");
			System.out.print("The probabilities after normalization: ");
			for (int k = 0; k < listOfExamples.size(); k++) {
				System.out.print(listOfExamples.get(k).prob + " ");
			}
			System.out.print("\nThe boosted classifier: ");
			printBoostedClassifier(boostedClassifier);
			System.out.println("The error of the boosted classifier: " + localClassifier.fractionalError);
			System.out.println("The bound on Et: " + bound + "\n\n");
		}
	}

	/*
	 * updates the hypothesized classification in the given list of examples
	 * using the hypothesis obtained
	 */
	private static void updateClassicationForHypotheis(ArrayList<example> listOfExamples, hypothesis h1) {
		// TODO Auto-generated method stub
		for (int i = 0; i < listOfExamples.size(); i++) {
			if (listOfExamples.get(i).x < h1.threshold) {
				if (h1.left) {
					listOfExamples.get(i).h = 1;
				} else {
					listOfExamples.get(i).h = -1;
				}
			} else {
				if (h1.left) {
					listOfExamples.get(i).h = -1;
				} else {
					listOfExamples.get(i).h = 1;
				}
			}
		}

	}

}
