
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

class classifier {
	hypothesis h1;
	double alpha;
	double fractionalError;
	double boundOnEt;
	double Ctplus;
	double Ctminus;
	double Gvalue;
	double Zvalue;

}

class example {
	double x;
	int y;
	double prob;
	int num;
	int h;
	double f;
}

class hypothesis {
	double threshold;
	boolean left;
	double Gvalue;
	double rightlyClassifiedPositives;
	double rightlyClassifiedNegatives;
	double wronglyClassifiedPositives;
	double wronglyClassifiedNegatives;
}

public class RealAdaBoosting {

	static int noOfIterations;
	static double epsilon;

	static ArrayList<example> readInput() throws FileNotFoundException {
		Scanner in = new Scanner(new File("adaboost-5.txt"));
		noOfIterations = in.nextInt();
		int noOfExamples = in.nextInt();
		epsilon = in.nextDouble();
		ArrayList<example> listOfExamples = new ArrayList<>();
		for (int i = 0; i < noOfExamples; i++) {
			example ex = new example();
			ex.x = in.nextDouble();
			ex.num = i;
			listOfExamples.add(ex);
		}
		for (int i = 0; i < noOfExamples; i++) {
			listOfExamples.get(i).y = in.nextInt();
		}
		for (int i = 0; i < noOfExamples; i++) {
			listOfExamples.get(i).prob = in.nextDouble();
		}
		in.close();
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
		double rightlyClassifiedPositives, rightlyClassifiedNegatives, wronglyClassifiedPositives,
				wronglyClassifiedNegatives;
		double minOfAllThresholds = Double.MAX_VALUE;
		double Grcp = 0;
		double Grcn = 0;
		double Gwcp = 0;
		double Gwcn = 0;

		int minOfAllEx = 0;
		boolean globalLeft = false;
		for (int i = 0; i <= listOfExamples.size() - 1; i++) {

			boolean leftPos = true;
			boolean right;
			boolean left = right = false;
			int noOfCases = 2;
			double Gvalue = 0;
			double minProbSum = Double.MAX_VALUE;
			double rcp = 0;
			double rcn = 0;
			double wcp = 0;
			double wcn = 0;
			while (noOfCases != 0) {
				rightlyClassifiedPositives = 0.0;
				rightlyClassifiedNegatives = 0.0;
				wronglyClassifiedPositives = 0.0;
				wronglyClassifiedNegatives = 0.0;
				Gvalue = 0.0;
				if (noOfCases != 2) {
					leftPos = false;
				}
				// counting the probabilities of the examples both rightly
				// wrongly classified below the threshold
				for (int j = 0; j < i; j++) {
					if (leftPos) {
						// if lefts are positive and the classification of the
						// ex in the left fold is not 1 then it is misclassified
						if (listOfExamples.get(j).y != 1) {
							wronglyClassifiedNegatives = wronglyClassifiedNegatives + listOfExamples.get(j).prob;
						} // if lefts are positive and the classification of the
							// ex is positive it is correctly classified
						else {
							rightlyClassifiedPositives = rightlyClassifiedPositives + listOfExamples.get(j).prob;
						}
					} else {
						// if rights are positive and the classification of the
						// ex
						// in the left fold is not -1 then it is misclassified
						if (listOfExamples.get(j).y != -1) {
							wronglyClassifiedPositives = wronglyClassifiedPositives + listOfExamples.get(j).prob;
						} else {
							rightlyClassifiedNegatives = rightlyClassifiedNegatives + listOfExamples.get(j).prob;
						}
					}
				}
				// counting probabilities for examples both rightly and wrongly
				// classified above the threshold
				for (int j = listOfExamples.size() - 1; j >= i; j--) {
					if (leftPos) {
						// if rights are positive and the classification of the
						// ex in the right fold is not 1 then it is
						// misclassified
						if (listOfExamples.get(j).y == 1) {
							wronglyClassifiedPositives = wronglyClassifiedPositives + listOfExamples.get(j).prob;
						} else if (listOfExamples.get(j).y == -1) {
							rightlyClassifiedNegatives = rightlyClassifiedNegatives + listOfExamples.get(j).prob;
						}
					} else {
						// if lefts are positive and the classification of the
						// ex
						// in the right fold is not -1 then it is misclassified
						if (listOfExamples.get(j).y != -1) {
							rightlyClassifiedPositives = rightlyClassifiedPositives + listOfExamples.get(j).prob;
						} else {
							wronglyClassifiedNegatives = wronglyClassifiedNegatives + listOfExamples.get(j).prob;
						}
					}
				}
				// System.out.println("for case " + noOfCases + " "+
				// rightlyClassifiedPositives + " " + rightlyClassifiedNegatives
				// + " " + wronglyClassifiedPositives + " " +
				// wronglyClassifiedNegatives );
				Gvalue = Math.sqrt(rightlyClassifiedPositives * wronglyClassifiedNegatives)
						+ Math.sqrt(rightlyClassifiedNegatives * wronglyClassifiedPositives);
				// if(i == 3)
				// System.out.println( " Gvalue:" + Gvalue);
				if (minProbSum > Gvalue) {
					minProbSum = Gvalue;
					rcp = rightlyClassifiedPositives;
					rcn = rightlyClassifiedNegatives;
					wcp = wronglyClassifiedPositives;
					wcn = wronglyClassifiedNegatives;
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
				Grcp = rcp;
				Grcn = rcn;
				Gwcp = wcp;
				Gwcn = wcn;
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
		/*
		 * hCurrent.threshold = (listOfExamples.get(minOfAllEx).x +
		 * listOfExamples .get(minOfAllEx - 1).x) / 2;
		 */hCurrent.left = globalLeft;
		hCurrent.Gvalue = minOfAllThresholds;
		hCurrent.rightlyClassifiedPositives = Grcp;
		hCurrent.rightlyClassifiedNegatives = Grcn;
		hCurrent.wronglyClassifiedPositives = Gwcp;
		hCurrent.wronglyClassifiedNegatives = Gwcn;
		return hCurrent;
	}

	static/*
			 * function to print the boosted classifier
			 */
	void printBoostedClassifier(ArrayList<classifier> boostedClassifier) {

		for (int i = 0; i < boostedClassifier.size(); i++) {
			if (boostedClassifier.get(i).h1.left) {
				System.out.print(" ( x < " + boostedClassifier.get(i).h1.threshold + " )");
			} else {
				System.out.print(" ( x > " + boostedClassifier.get(i).h1.threshold + " )");

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
		// double[] alphas = new double[noOfIter];
		double[] Zt = new double[noOfIterations];
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
		for (int i = 0; i < noOfIterations; i++) {
			System.out.println("Iteration: " + i);
			// printList(listOfExamples);
			h1 = getHypothesis(listOfExamples);
			// System.out.println(h1.Gvalue);
			if (h1.left) {
				System.out.println("The selected weak classifier Ht: x < " + h1.threshold);
			} else {
				System.out.println("The selected weak classifier Ht: x > " + h1.threshold);
			}
			System.out.println("The G error value of Ht: " + h1.Gvalue);
			updateClassicationForHypotheis(listOfExamples, h1);
			/*
			 * if the classifier is not week we can break and run the algorithm
			 * again
			 */
			if (h1.Gvalue == 0) {
				System.out.println("classifier is not week Run the algo again");
				break;
			}
			// computing the Zt value
			Zt[i] = 2 * h1.Gvalue;

			/*
			 * System.out.println(h1.rightlyClassifiedPositives + " " +
			 * h1.rightlyClassifiedNegatives + " " +
			 * h1.wronglyClassifiedPositives + " " +
			 * h1.wronglyClassifiedNegatives);
			 */// computing the Ct+ and Ct- values
				// epsilon value is hard coded this needs to be changed
			double Ctplus, Ctminus;
			Ctplus = (0.5)
					* Math.log((h1.rightlyClassifiedPositives + epsilon) / (h1.wronglyClassifiedNegatives + epsilon));
			Ctminus = (0.5)
					* Math.log((h1.wronglyClassifiedPositives + epsilon) / (h1.rightlyClassifiedNegatives + epsilon));
			System.out.println("The weights Ct+,Ct-: " + Ctplus + "," + Ctminus);
			System.out.println("The probabilities normalization factor Zt: " + Zt[i]);
			// computing the updated probabilities
			wrongCount = 0;
			for (int j = 0; j < listOfExamples.size(); j++) {
				if (listOfExamples.get(j).h == 1) {
					listOfExamples.get(j).prob = (listOfExamples.get(j).prob
							* Math.pow(Math.E, -(listOfExamples.get(j).y * Ctplus))) / Zt[i];
				} else {
					listOfExamples.get(j).prob = (listOfExamples.get(j).prob
							* Math.pow(Math.E, -(listOfExamples.get(j).y * Ctminus))) / Zt[i];
				}
				/*
				 * if (listOfExamples.get(j).y != listOfExamples.get(j).h) {
				 * wrongCount++; }
				 */
			}
			System.out.print("The probabilities after normalization: ");
			for (int k = 0; k < listOfExamples.size(); k++) {
				System.out.print(listOfExamples.get(k).prob + ",");
			}
			System.out.println();
			// adding h1,alpha from each iteration to the boostedClassifier
			classifier localClassifier = new classifier();
			localClassifier.h1 = h1;
			localClassifier.Ctplus = Ctplus;
			localClassifier.Ctminus = Ctminus;
			localClassifier.Zvalue = Zt[i];
			// localClassifier.alpha = alphas[i];
			// printBoostedClassifier(boostedClassifier);
			System.out.print("The values ft(xi) for each one of the examples: ");
			for (int k = 0; k < listOfExamples.size(); k++) {
				if (listOfExamples.get(k).h == -1) {
					listOfExamples.get(k).f = listOfExamples.get(k).f + Ctminus;
				} else {
					listOfExamples.get(k).f = listOfExamples.get(k).f + Ctplus;
					// System.out.print(Ctplus + " ");
				}
			}
			for (int k = 0; k < listOfExamples.size(); k++) {
				System.out.print(listOfExamples.get(k).f + ", ");
				if (listOfExamples.get(k).f < 0 && listOfExamples.get(k).y != -1) {
					wrongCount++;
				} else if (listOfExamples.get(k).f >= 0 && listOfExamples.get(k).y != 1) {
					wrongCount++;
				}
				// System.out.println(wrongCount);
			}
			// System.out.println();
			// System.out.println(wrongCount);
			// computing fractional error of the boosted classifier Et
			localClassifier.fractionalError = wrongCount / listOfExamples.size();
			bound = 1;
			// computing the bound on Et
			for (int k = 0; k <= i; k++) {
				bound = bound * Zt[k];
			}
			localClassifier.boundOnEt = bound;
			boostedClassifier.add(localClassifier);

			// System.out.println();
			System.out.print("\nThe error of the boosted classifier Et: " + localClassifier.fractionalError);
			System.out.print("The bound on Et: " + bound + "\n\n");

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
