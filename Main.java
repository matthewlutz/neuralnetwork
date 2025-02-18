//Matthew Lutz CS457, 12/02/2024


import java.io.*;
import java.util.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Main {
	public static int debugIndex = 1;
	
	public static void main(String[] args) {
		String filename = null; //filename
		int hiddenLayers = 0; //hiddenLayers
		double alpha = .01; //learning rate
		int epochLimit = 1000; //epochlimit
		int batchSize = 1; //batchsize
		double lambda = 0.0; //lambda
		boolean randomize = false; //randomization
		double weightInit = .1; //weight initialization
		int verbosity = 1; //verbosity
		List<Integer> hiddenLayerSizes = new ArrayList<>();
		
		try {
			for(int i = 0; i < args.length; i++) {
				switch(args[i]) {
					case "-f":
	                    if (++i < args.length) {
	                  	  filename = args[i];
	                      //System.out.println(filename);
	                    }
	                    else throw new IllegalArgumentException("Expected filename after -f");
	                    break;
					case "-h":
	                    if (++i < args.length) {
	                  	  hiddenLayers = Integer.parseInt(args[i]);
	                  	  hiddenLayerSizes = new ArrayList<>();
	                  	  for(int j =0; j < hiddenLayers; j++) {
	                  		  if(++i < args.length) {
		                  		  int x = Integer.parseInt(args[i]);
		                  		  hiddenLayerSizes.add(x);
	                  		  }else {
	                              throw new IllegalArgumentException("Expected a size input for hidden layer " + (j + 1));
	                  		  }
	                  	  }
	                    }
	                    else throw new IllegalArgumentException("Expected hiddenLayers after -h");
	                    break;
					case "-a":
	                    if (++i < args.length) {
	                    	  alpha = Double.parseDouble(args[i]);
	                      }
	                      else throw new IllegalArgumentException("Expected alpha after -a");
	                      break;
					case "-e":
	                    if (++i < args.length) {
	                  	  epochLimit = Integer.parseInt(args[i]);
	                    }
	                    else throw new IllegalArgumentException("Expected epochLimit after -e");
	                    break;
					case "-m":
	                    if (++i < args.length) {
	                  	  batchSize = Integer.parseInt(args[i]);
	                    }
	                    else throw new IllegalArgumentException("Expected batchsize after -m");
	                    break;     
					case "-l":
	                    if (++i < args.length) {
	                  	  lambda = Double.parseDouble(args[i]);
	                    }
	                    else throw new IllegalArgumentException("Expected lambda after -l");
	                    break; 
					case "-r":
						randomize = true;
						break;
					case "-w":
	                    if (++i < args.length) {
	                  	  weightInit = Double.parseDouble(args[i]);
	                    }
	                    else throw new IllegalArgumentException("Expected weightinit after -w");
	                    break;
					case "-v":
	                    if (++i < args.length) {
	                  	  verbosity = Integer.parseInt(args[i]);
	                    }
	                    else throw new IllegalArgumentException("Expected verbosity after -v");
	                    break;
				}
			}
			
		}catch (NumberFormatException e) {
	        System.err.println("Error parsing numerical argument: " + e.getMessage());
	        return;
	    } catch (IllegalArgumentException e) {
	        System.err.println("Argument error: " + e.getMessage());
	        return;
	    }
	
	    if (filename == null) {
	        System.err.println("Filename must be specified with -f.");
	        return;
	    }
	    
	    if(filename != null) {
	    	List<String> dataLines = readDataFromFile(filename); //read file
	    	System.out.println("* Reading " + filename);
	    	
	    	List<Double[]> featureList = new ArrayList<>();
	    	List<int[]> targetList = new ArrayList<>();
	    	parseLines(dataLines, featureList, targetList);	//parse each line
	    	
	    	Map<String, List<DataPair>> dataMap = prepareData(featureList, targetList, randomize); //prepare data
	    	List<DataPair> trainingSet = dataMap.get("trainingPart"); //prepare data
	    	List<DataPair> validationSet = dataMap.get("validationPart"); //prepare data
	    	System.out.println("* Doing train/validation split");
	    	
	    	System.out.println("* Scaling features");
	    	scaleFeatures(trainingSet, validationSet, verbosity); //scale features
	    	
	    	//get input and output size
	    	int inputSize = trainingSet.get(0).features.length;
	    	int outputSize = trainingSet.get(0).targets.length;
	    	
	    	if (hiddenLayers > 0) {
	            for (int i = 0; i < hiddenLayers; i++) {
	                hiddenLayerSizes.add(hiddenLayerSizes.get(i));
	            }
	        }
	    	
	    	//build the network
	    	System.out.println("* Building network");
	    	NeuralNetwork network = new NeuralNetwork(inputSize, outputSize, weightInit, alpha, hiddenLayerSizes);
	    	printLayerSizes(network);
	    	//System.out.println("network built successfully");
	    	
	    	System.out.println("* Training network (using " + epochLimit + " examples)");
	    	trainNetwork(network, trainingSet, epochLimit, batchSize, alpha, lambda, verbosity); //trains the training set
	    	validateNetwork(network, validationSet, epochLimit, batchSize, verbosity); //validates with validation set
	    	
	    	
	    	//output
	    	double trainAcc = evaluateAccuracy(network, trainingSet, verbosity);
	    	double valAcc = evaluateAccuracy(network, validationSet, verbosity);
	    	System.out.println("* Evaluating Accuracy");
	    	System.out.printf("  TrainAcc: %.6f%n", trainAcc);
	    	System.out.printf("  ValidAcc: %.6f%n", valAcc);
	    }
	}
	
	//this method reads the data from the input file, skipping lines that are empty or start with '#', and adding them to an array list of strings that holds each line.
	private static List<String> readDataFromFile(String filename){
		List<String> dataLines = new ArrayList<>();
		try (BufferedReader br = new BufferedReader(new FileReader(filename))){
			String line;
			while((line = br.readLine()) != null) {
				line = line.trim();
				if(line.isEmpty() || line.startsWith("#") ) {
					continue;
				}
				dataLines.add(line);
			}
		}catch(FileNotFoundException e) {
	        System.err.println("File not found: " + filename);
	    } catch (IOException e) {
	        System.err.println("Error reading file: " + filename);
	    }
		return dataLines;
	}
	
	//this method parses each line in datalines, and adds a feature array to the feature list, and target array to the target list
	private static void parseLines(List<String> dataLines, List<Double[]> featureList, List<int[]> targetList){
		//ex line format (0.665 0.790) (0 1 0)
		for(String line : dataLines) {
			//System.out.println(line);
			String[] parts = line.split("\\)\\s*\\(");
			if(parts.length > 2) { //invalid line input format 
				System.err.println("Invalid data format: " + line);
	            continue;
			}
			
			//get rid of the remaining parenthesis
			String featurePart = parts[0].replace("(", "").trim();
			String targetPart = parts[1].replace(")", "").trim();
			
			//parse features
			String[] featureTokens = featurePart.split("\\s+");
			Double[] features = new Double[featureTokens.length];
			for(int i =0; i < featureTokens.length; i++) {
				try {
					features[i] = Double.parseDouble(featureTokens[i]);
				}catch (NumberFormatException e) {
	                System.err.println("invalid feature value: " + featureTokens[i]);
	                continue;
	            }
			}
			
			//parse targets
			String[] targetTokens = targetPart.split("\\s+");
			int[] targets = new int[targetTokens.length];
			for(int i =0; i < targetTokens.length; i++) {
				try {
					targets[i] = Integer.parseInt(targetTokens[i]);
				}catch (NumberFormatException e) {
	                System.err.println("invalid target value: " + targetTokens[i]);
	                continue;
	            }
			}
			
			featureList.add(features);
			targetList.add(targets);
		}

	}
	
	
	//this method prepares the data. 80% in training set and 20% in validation set
	private static Map<String, List<DataPair>> prepareData(List<Double[]> featureList, List<int[]> targetList, boolean randomize) {
		//combine the features and targets for a line
		List<DataPair> dataPairs = new ArrayList<>();
		for(int i =0; i < featureList.size(); i++) {
			dataPairs.add(new DataPair(featureList.get(i), targetList.get(i)));
		}
		
		//if randomize flag is specified, randomize the data before separating into training and validation sets
		if(randomize) Collections.shuffle(dataPairs);
		
		//splitting the dataPairs 80/20, so the index is at .8 of the dataPairs list
		int splitIndex = (int) (dataPairs.size() * .8);
		List<DataPair> trainingSet = dataPairs.subList(0, splitIndex);
		List<DataPair> validationSet = dataPairs.subList(splitIndex, dataPairs.size());
		
		Map<String, List<DataPair>> resultMap = new HashMap<>();
		resultMap.put("trainingPart", trainingSet);
		resultMap.put("validationPart", validationSet);
		return resultMap;
	
	}
	
	//this method scales the features from -1 to 1
	private static void scaleFeatures(List<DataPair> trainingSet, List<DataPair> validationSet, int verbosity) {
		//make sure sets aren't empty
		if(trainingSet.isEmpty() || validationSet.isEmpty()) {
			System.out.println("sets are empty and can't be scaled;");
			return;
		}
		
		//all datapoints have the same amount of features, so can just use any set to get length
		int length = trainingSet.get(0).features.length;
		double[] minVals = new double[length];
		double[] maxVals = new double[length];
		//fill the arrays with MAX_VALUE to make sure that any value encountered in the feature list fits
		Arrays.fill(minVals, Double.MAX_VALUE);
		Arrays.fill(maxVals, -Double.MAX_VALUE);
		
		//find the min and max values for each feature
		for(DataPair data : trainingSet) {
			Double[] features = data.features;
			for(int j =0; j < features.length; j++) {
				minVals[j] = Math.min(minVals[j], features[j]);
				maxVals[j] = Math.max(maxVals[j], features[j]);
			}
		}
		
		if(verbosity >= 2) {
			System.out.println("  * Min/max values on training set:");
		    for (int i = 0; i < length; i++) {
		        System.out.printf("    Feature %d: %.3f, %.3f%n", i + 1, minVals[i], maxVals[i]);
		    }
		}
		
		
		//normalization algorithm for training set
		for(DataPair data : trainingSet) {
			Double[] features = data.features;
			for(int j = 0; j < features.length; j++) {
				//make sure that min and max arent equal to avoid dividing by zero
				if(minVals[j] == maxVals[j]) {
					features[j] = -1.0;
				}else {
					features[j] = -1 + 2 * ((features[j] - minVals[j]) / (maxVals[j] - minVals[j]));
				}
			}
		}
		
		//normalization alg for validation set
		for(DataPair data : validationSet) {
			Double[] features = data.features;
			for(int j = 0; j < features.length; j++) {
				//make sure that min and max arent equal to avoid dividing by zero
				if(minVals[j] == maxVals[j]) {
					features[j] = -1.0;
				}else {
					features[j] = -1 + 2 * ((features[j] - minVals[j]) / (maxVals[j] - minVals[j]));
				}
			}
		}	
	}
	
	
	//this method trains the network
	public static void trainNetwork(NeuralNetwork network, List<DataPair> trainingSet, int epochLimit, int batchSize, double alpha, double lambda, int verbosity) {
	    System.out.println("  * Beginning mini-batch gradient descent");
	    if(verbosity >= 2) {
			System.out.printf("    (batchSize=%d, epochLimit=%d, learningRate=%.4f, lambda=%.4f)%n", batchSize, epochLimit, alpha, lambda);
			if(verbosity >= 3) {
				double initialCost = calculateCost(network, trainingSet, lambda, verbosity);
				double initialAcc = evaluateAccuracy(network, trainingSet, verbosity);
				System.out.printf("    Initial model with random weights : Cost = %.6f; Loss = %.6f; Acc = %.4f%n", initialCost, initialCost, initialAcc);

			}
	    }
		
		
	    long startTime = System.currentTimeMillis();
	    int totalIterations = 0;
	    
	    double previousCost = Double.MAX_VALUE; //init to a large value
	    String stopCondition = "Epoch Limit"; 
	    
		for(int epoch = 0; epoch < epochLimit; epoch++) {
			Collections.shuffle(trainingSet); 
			
			for(int i = 0; i < trainingSet.size(); i+= batchSize) {
				int end = Math.min(i + batchSize, trainingSet.size());
	            for (int j = i; j < end; j++) {
	                DataPair data = trainingSet.get(j);
	                network.forwardPropagation(data.features, verbosity, 1, data.targets); //forward propagation
	                network.backPropagation(data.targets, verbosity, 1, data.features); //back propagation
	                totalIterations++;
	            }
			}
			
			if ((epoch ) % 100 == 0 || epoch == epochLimit - 1 && verbosity == 3) {
	            double currentCost = calculateCost(network, trainingSet, lambda, verbosity);
	            double currentAcc = evaluateAccuracy(network, trainingSet, verbosity);
	            System.out.printf("    After %d epochs (%d iter.): Cost = %.6f; Loss = %.6f; Acc = %.4f%n", epoch, totalIterations, currentCost, currentCost, currentAcc);
	        }
			
			double currentCost = calculateCost(network, trainingSet, lambda, verbosity);
						
			if (Math.abs(previousCost - currentCost) < .001) {
	            stopCondition = "Convergence";
	            break;
	        }
	       
	        previousCost = currentCost;
		}
		
		long endTime = System.currentTimeMillis();
	    double trainingTime = (endTime - startTime) / 1000.0;

	    System.out.println("  * Done with fitting!");
	    System.out.printf("    Training took %.3f seconds, %d epochs, %d iterations (%.6fms / iteration)%n",
	            trainingTime, epochLimit, totalIterations, (trainingTime * 1000) / totalIterations);

	    System.out.println("    GD Stop condition: " + stopCondition);
	}
	
	//this method validates the network, using the validation set on the network after the training set has ran
	public static void validateNetwork(NeuralNetwork network, List<DataPair> validationSet, int epochLimit, int batchSize, int verbosity) {
			int correct = 0;
			
			for(DataPair data : validationSet) {
				double[] outputs = network.forwardPropagation(data.features, verbosity, 0, null);
		        int predicted = getPredicted(outputs);
		        int actual = getActual(data.targets);
		        
		        if (predicted == actual) {
		            correct++;
		        }
			}
			
			double accuracy = (double) correct / validationSet.size();
		    //System.out.println("test accuracy: " + (accuracy * 100) + "%");
	}
	
	//this method is used to calculate the accuracy for training and validation sets
	public static double evaluateAccuracy(NeuralNetwork network, List<DataPair> dataset, int verbosity) {
		int correct = 0;

	    for (DataPair data : dataset) {
	        double[] outputs = network.forwardPropagation(data.features, verbosity, 0, null);
	        int predicted = getPredicted(outputs);
	        int actual = getActual(data.targets);
	        if (predicted == actual) {
	            correct++;
	        }
	    }

	    return (double) correct / dataset.size();
	}
	
	//helper method for validate network method that finds the predicted index in outputs
	public static int getPredicted(double[] outputs) {
		int prediction = 0;
		for(int i = 1; i < outputs.length; i++) {
			if(outputs[i] > outputs[prediction]) {
				prediction = i;
			}
		}
		return prediction;
	}
	
	//helper method for validate method that gets the actual index in outputs
	public static int getActual(int[] targets) {
		for(int i =0; i < targets.length; i++) {
			if(targets[i] == 1) {
				return i;
			}
		}
		return -1; //safety case
	}
	
	public static void printLayerSizes(NeuralNetwork network) {
		System.out.println("  * Layer sizes (excluding bias neuron(s)):");
	    for (int i = 0; i < network.layers.size(); i++) {
	        Layer layer = network.layers.get(i);
	        int numNeurons = (int) layer.getNeurons().stream().filter(n -> !n.isBias()).count();
	        System.out.printf("    Layer %d (%s): %d%n", i + 1, (i == 0 ? "input" : (i == network.layers.size() - 1 ? "output" : "hidden")), numNeurons);
	    }
	}
	
	//this method is used to calculate the cost
	private static double calculateCost(NeuralNetwork network, List<DataPair> dataset, double lambda, int verbosity) {
	    double totalCost = 0.0;

	    for (DataPair data : dataset) {
	        double[] outputs = network.forwardPropagation(data.features, verbosity, 0, null);
	        int[] targets = data.targets;
	    
	        for (int i = 0; i < outputs.length; i++) {
	            totalCost -= targets[i] * Math.log(outputs[i] + 1e-15);
	        }
	    }

	    double regularization = 0.0;
	    for (Layer layer : network.layers) {
	        for (Neuron neuron : layer.getNeurons()) {
	            if (!neuron.isBias()) {
	                for (double weight : neuron.getWeights()) {
	                    regularization += weight * weight;
	                }
	            }
	        }
	    }
	    totalCost += (lambda / 2.0) * regularization;

	    return totalCost / dataset.size();
	}
	
	
	
	//this class is for joining data together from the feature list and target list
	private static class DataPair {
		Double[] features;
	    int[] targets;
	    
	    //constructor
	    DataPair(Double[] features, int[] targets){
	    	this.features = features;
	    	this.targets = targets;
	    }
	}
	
	//this class is the networks structure and behavior 
	private static class NeuralNetwork{
		private List<Layer>  layers;
		private double epsilon;
		private double lambda;
		
		//constructor
		public NeuralNetwork(int inputSize, int outputSize, double epsilon, double lambda, List<Integer> hiddenLayerSizes ) {
			this.epsilon = epsilon;
			this.lambda = lambda;
			this.layers = new ArrayList<>();
			layers.add(new Layer(inputSize, true)); //add input layer
			
			//add hiddenLayers
			for(int size : hiddenLayerSizes) {
				layers.add(new Layer(size, true));
			}
			
			layers.add(new Layer(outputSize, false)); //output layer
			
			initWeights();		
			
		}
		
		//use the layer class to get the current and next layer and then use the layer initWeights method to init the weights
		public void initWeights() {
			for (int i = 0; i < layers.size(); i++) {
		        Layer currentLayer = layers.get(i);
		        currentLayer.initializeLayerNeurons(epsilon); // Initialize neurons
		    }

		    for (int i = 0; i < layers.size() - 1; i++) {
		        Layer currentLayer = layers.get(i);
		        Layer nextLayer = layers.get(i + 1);
		        currentLayer.initWeights(nextLayer, epsilon);
		    }
		}
		
		public double[] forwardPropagation(Double[] inputValues, int verbosity, int debug, int[] targetValues) {
			//get the input layer
			Layer inputLayer = layers.get(0);
			//set the input values to the input neurons 
			//System.out.println("input values: " + Arrays.toString(inputValues));
			for(int i =0; i < inputValues.length; i++) {
				inputLayer.getNeurons().get(i).setOutput(inputValues[i]);
		        //System.out.println("output for input neuron " + i + " to " + inputValues[i]);
			}
			
			//int debug is a flag to run v level 4 output
			if(verbosity == 4 && debug == 1) {
				System.out.println("    * Forward propogation on example " + debugIndex);
			    System.out.print("      Layer 1 (input):    a_j: 1.000 "); //bias neuron
			    for (Neuron neuron : inputLayer.getNeurons()) {
			        System.out.printf("%.3f ", neuron.getOutput());
			    }
			    System.out.println();
			}
			
			//go through rest of layers
			for(int i = 1; i < layers.size(); i++) {
				//start by grabbing the previous and current layers 
				Layer prevLayer = layers.get(i-1);
				Layer currLayer = layers.get(i);
				
				if (verbosity == 4 && debug == 1) {
		            System.out.print("      Layer " + (i + 1) + ":");
		            System.out.print("      in_j: ");
		        }
		        //System.out.println("going through layer " + i);

				//go through neurons for the current layer
				for(int j = 0; j < currLayer.getNeurons().size(); j++) {
					
					Neuron neuron = currLayer.getNeurons().get(j);
					double sum = 0.0;
					//go through the previous layer
					for(Neuron prevNeuron : prevLayer.getNeurons()) {
						//skip bias neurons
						if (prevNeuron.isBias()) {
					        continue;
					    }
		                //System.out.println("weight for connection: prevNeuron -> neuron " + j);

						double weight = prevNeuron.getWeights()[j];
						sum += prevNeuron.getOutput() * weight;
						
					}
					if (verbosity == 4 && debug == 1) {
		                System.out.printf("%.3f ", sum);
		            }
		            neuron.applyActivation(sum);
				}
				if (verbosity == 4 && debug == 1) {
		            System.out.println();
		            System.out.print("                    a_j: ");
		            for (Neuron neuron : currLayer.getNeurons()) {
		                System.out.printf("%.3f ", neuron.getOutput());
		            }
		            System.out.println();
		        }
			}
			
			//get output values from layer
			Layer outputLayer = layers.get(layers.size() - 1);
		    double[] outputValues = new double[outputLayer.getNeurons().size()];
		    for(int i = 0; i < outputValues.length; i++) {
		    	outputValues[i] = outputLayer.getNeurons().get(i).getOutput();
		        //System.out.println("Output neuron " + i + " final value: " + outputValues[i]);

		    }
		    
		    
		    if (verbosity == 4 && debug == 1) {
		        System.out.print("          examples actual y: ");
		        for (double target : targetValues) {
		            System.out.printf("%.3f ", target);
		        }
		        System.out.println();
		        debugIndex++;
		    }
		    
		    return outputValues;
						
		}
		
		public void backPropagation(int[] targetValues, int verbosity, int debug, Double[] inputValues) {
			//get output layer
			Layer outputLayer = layers.get(layers.size()-1);
			//double sum = 0.0;
			for (int i = 0; i < outputLayer.getNeurons().size(); i++) {
		        Neuron neuron = outputLayer.getNeurons().get(i);
		        double output = neuron.getOutput();
		        double error = (targetValues[i] - output) * neuron.activationDerivative(output);
		        neuron.setError(error);
		    }
			
			
			if (verbosity == 4 && debug == 1) {
		        System.out.println("    * Backward propagation on example " + debugIndex);
		        System.out.print("      Layer " + layers.size() + " (output):      Delta_j: ");
		        for (Neuron neuron : outputLayer.getNeurons()) {
		            System.out.printf("%.3f ", neuron.getError());
		        }
		        System.out.println();
		    }
		 

			
			//backpropagate through rest of the layers
			for(int layerIndex = layers.size() - 2; layerIndex >= 1; layerIndex--) {
				//grab layers				
				Layer currLayer = layers.get(layerIndex);
				Layer nextLayer = layers.get(layerIndex + 1);
				
				if (verbosity == 4 && debug == 1) {
		            System.out.print("      Layer " + (layerIndex + 1) + " (hidden):      Delta_j: ");
		        }

				//go through current layer of neurons 
				for(int i =0; i < currLayer.getNeurons().size(); i++) {
					Neuron neuron = currLayer.getNeurons().get(i);
					if (neuron.isBias()) {
				        //System.out.println("skipping bias in backpropagation for layer " + layerIndex);
				        continue;
				    }
					double sum = 0.0;
					
					//go through next layer of neurons
					for(int j = 0; j < nextLayer.getNeurons().size(); j++) {						
						Neuron nextNeuron = nextLayer.getNeurons().get(j);
						if(nextNeuron.isBias()) continue;
						double[] nextWeights = neuron.getWeights();
						//System.out.println(neuron.bias);
						//System.out.println("test" + nextNeuron.getWeights()[i]);
						if (nextWeights.length > i) {
		                    sum += nextNeuron.getError() * nextWeights[i];
		                }
					}
					
					double output = neuron.getOutput();
			        double error = sum * neuron.activationDerivative(output); //logistic derivative
			        neuron.setError(error);
						
			        //print delta_j for v level 4
		            if (verbosity == 4 && debug == 1) {
		                System.out.printf("%.3f ", error);
		            }
		      
				}
				if(verbosity == 4 && debug == 1) {
					System.out.println();
				}				
			}
			
			//update weights 
			//skip input layer
			for(int layerIndex = 1; layerIndex < layers.size(); layerIndex++) {
				Layer currLayer = layers.get(layerIndex);
				Layer prevLayer = layers.get(layerIndex - 1);
				
				for(int i = 0; i < currLayer.getNeurons().size(); i++) {
			        Neuron neuron = currLayer.getNeurons().get(i);
					if (neuron.isBias()) {
			            continue;
			        }
					double[] weights = neuron.getWeights();
			        for (int j = 0; j < prevLayer.getNeurons().size(); j++) {
			            Neuron prevNeuron = prevLayer.getNeurons().get(j);			            
			            if (j < weights.length) {
			                double gradientStep = epsilon * neuron.getError() * prevNeuron.getOutput();
			                weights[j] -= gradientStep;
			            }
			        }
				}
			}						
		}	
	}
	
	//this class represents 1 layer in the network
	private static class Layer{
		public List<Neuron> neurons;
		boolean bias;
		
		//constructor
		public Layer(int size, boolean bias) {
			this.neurons = new ArrayList<>();
			for(int i =0; i < size; i++) {
				neurons.add(new Neuron());
			}
			
			//handle the bias with a0 having a constant output of 1
			this.bias = bias;
			if(bias) {
				neurons.add(new Neuron(1.0)); 
			}
		}
		
		public void initWeights(Layer nextLayer, double epsilon) {
			Random rand = new Random();
			int edges = nextLayer.getNeurons().size();
			for(Neuron n : neurons) {				
				if (!n.isBias()) { //skip init if bias neuron
		            n.initWeights(edges, epsilon, rand);
		        }
		        //System.out.println("weights initialized for neuron with size: " + n.getWeights().length);
			}
		}
		
		public List<Neuron> getNeurons() {
			return neurons;
		}
		
		public void initializeLayerNeurons(double epsilon) {
		    for (Neuron neuron : neurons) {
		        if (!neuron.isBias()) {
		            neuron.initWeights(0, epsilon, new Random());
		        }
		    }
		}
		
	}
	
	//this class represents 1 neuron in the network
	private static class Neuron{
		private double bias;
		private double output;
		private double[] weights;
		private double error;
		
		//normal constructor 
		public Neuron() {
			this(0.0);
		}
		
		//constructor for bias 
		public Neuron(double bias) {
			this.bias = bias;
		}
		
		//method to initialize the weights 
		public void initWeights(int edges, double epsilon, Random rand) {
			if (edges <= 0) {
				weights = new double[0];
		        return;
		    }
			
			weights = new double[edges];
			for (int i = 0; i < edges; i++) {
	            weights[i] = (rand.nextDouble() * 2 * epsilon) - epsilon; //random weight initialization 

	        }			
			//System.out.println("initialized weights for neuron with " + edges + " outgoing edges. weights: " + Arrays.toString(weights));
		}
		
		//method to return weights
		public double[] getWeights() {
			if(weights == null) throw new IllegalStateException("Weights are not initialized for this neuron.");
			return weights;
		}
		
		//method to set the output
		public void setOutput(double val) {
			this.output = val;
		}
		
		//method to return the output
		public double getOutput() {
			return output;
		}
		
		//this method uses the standard logistic activation function to activate a processing neuron 
		public double activation(double input) {
			return ( 1 / (1 + Math.exp(-input)));
		}
		
		//this method is the derivative for the standard logistic activation function
		public double activationDerivative(double output) {
			return (output * (1 - output));
		}
		
		//this method computes the output using the activation method 
		public void applyActivation(double input) {
			this.output = activation(input);
		}
		
		//this method sets the error
		public void setError(double error) {
			this.error = error;
		}
		
		//this method returns the error
		public double getError() {
			return error;
		}
		
		public boolean isBias() {
			return bias == 1.0;
		}
		
	}

}

	

