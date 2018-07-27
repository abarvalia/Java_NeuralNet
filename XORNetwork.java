import java.util.Random;
import java.util.ArrayList;
import java.util.Collections;
import javax.swing.JOptionPane;

public class XORNetwork {
  
  //multilayer network to learn XOR gate
  
  public static void main(String[] args) {
     final double learningRate = 0.2;
     final double momentum = 0.85;
   
     double deltaOut, deltaHidden1, deltaHidden2, error, outputValue, in1h1_change, in1h2_change, in2h1_change, in2h2_change, inbh1_change, 
       inbh2_change, h1out_change, h2out_change, hbout_change;
     int trainingCycles;
    
     
     ArrayList<Double[]> trainingData = new ArrayList<Double[]>();
     trainingData.add(new Double[] {0.0, 0.0});
     trainingData.add(new Double[] {0.0, 1.0});
     trainingData.add(new Double[] {1.0, 0.0});
     trainingData.add(new Double[] {1.0, 1.0});
     
     ArrayList<Integer> indexValues = new ArrayList<Integer>();
     indexValues.add(0);
     indexValues.add(1);
     indexValues.add(2);
     indexValues.add(3);
     
     double[] desiredVals = new double[] {0.0, 1.0, 1.0, 0.0};
     
     while (true) {
     String input = JOptionPane.showInputDialog("Welcome to the logical XOR neural network. How many training cycles do you wish to run? (Type q to quit)");
     if (input.equals("q")) {
       break;
     }
     else {
       trainingCycles = Integer.parseInt(input);
     }
     
     //Instantiate two input neurons, one ouput neuron, one bias
     Neuron input1 = new Neuron("Input");
     Neuron input2 = new Neuron("Input");
     Neuron inputBias = new Neuron("Bias");
   
     Neuron hidden1 = new Neuron("Hidden");
     Neuron hidden2 = new Neuron("Hidden");
     Neuron hiddenBias = new Neuron("Bias");

     Neuron output = new Neuron("Output");
     
    //Instantiate six connections between each input neuron and each hidden layer neuron plus bias to two hidden layer neurons
     Connection in1h1 = new Connection(input1, hidden1, input1.setRandomWeight());
     Connection in1h2 = new Connection(input1, hidden2, input1.setRandomWeight());
     Connection in2h1 = new Connection(input2, hidden1, input1.setRandomWeight());
     Connection in2h2 = new Connection(input2, hidden2, input1.setRandomWeight());
     Connection inbh1 = new Connection(inputBias, hidden1, input1.setRandomWeight());
     Connection inbh2 = new Connection(inputBias, hidden2, input1.setRandomWeight());
     //Set Hidden Layer Connections Array
     hidden1.setConnectionsArray(in1h1);
     hidden1.setConnectionsArray(in2h1);
     hidden1.setConnectionsArray(inbh1);
     hidden2.setConnectionsArray(in1h2);
     hidden2.setConnectionsArray(in2h2);
     hidden2.setConnectionsArray(inbh2);
     //Instantiate three connections between hidden layer neurons and output neuron, as well as bias
     Connection h1out = new Connection(hidden1, output, hidden1.setRandomWeight());
     Connection h2out = new Connection(hidden2, output, hidden1.setRandomWeight());
     Connection hbout = new Connection(hiddenBias, output, hidden1.setRandomWeight());
     //Set Output Layer Connections Array
     output.setConnectionsArray(h1out);
     output.setConnectionsArray(h2out);
     output.setConnectionsArray(hbout);
     
     System.out.println("Training the network...");
     
     for (int cycle = 0; cycle < trainingCycles; cycle++) {
       Collections.shuffle(indexValues);
       // perform one pass through the training data
       for (Integer index : indexValues) {          
         Double[] data = trainingData.get(index);
         double target = desiredVals[index];
       
         input1.setInputVal(data[0]);
         input2.setInputVal(data[1]);
         inputBias.setInputVal(1);
         hiddenBias.setInputVal(1);
         //hidden layer computes values
         hidden1.computeValue();
         hidden2.computeValue();
         //output computes value 
         outputValue = output.computeValue();

         //compute error
         error = target - outputValue;
         //compute delta value of output & hidden neurons
         deltaOut = outputValue * (1-outputValue) * error;
         deltaHidden1 = deltaOut * h1out.getWeight() * hidden1.getValue() * (1 - hidden1.getValue());
         deltaHidden2 = deltaOut * h2out.getWeight() * hidden2.getValue() * (1 - hidden2.getValue());
         
         //compute the weight change for each of the connections to the hidden layer
         in1h1_change = ((learningRate) * (input1.getValue()) * deltaHidden1) + (momentum * in1h1.getPrevChange());
         in1h2_change = ((learningRate) * (input1.getValue()) * deltaHidden2) + (momentum * in2h1.getPrevChange());
         in2h1_change = ((learningRate) * (input2.getValue()) * deltaHidden1) + (momentum * in2h1.getPrevChange());
         in2h2_change = ((learningRate) * (input2.getValue()) * deltaHidden2) + (momentum * in2h2.getPrevChange());
         inbh1_change = ((learningRate) * (inputBias.getValue()) * deltaHidden1) + (momentum * inbh1.getPrevChange());
         inbh2_change = ((learningRate) * (inputBias.getValue()) * deltaHidden2) + (momentum * inbh2.getPrevChange());
         //compute the weight change for each of the connections to the output layer
         h1out_change = ((learningRate) * (hidden1.getValue()) * deltaOut) + (momentum * h1out.getPrevChange());
         h2out_change = ((learningRate) * (hidden2.getValue()) * deltaOut) + (momentum * h2out.getPrevChange());
         hbout_change = ((learningRate) * (hiddenBias.getValue()) * deltaOut) + (momentum * hbout.getPrevChange());
         

         //update weights for connections to hidden layer:        
         in1h1.updateWeight(in1h1.getWeight() + in1h1_change);
         in1h2.updateWeight(in1h2.getWeight() + in1h2_change); 
         in2h1.updateWeight(in2h1.getWeight() + in2h1_change);
         in2h2.updateWeight(in2h2.getWeight() + in2h2_change);
         inbh1.updateWeight(inbh1.getWeight() + inbh1_change);
         inbh2.updateWeight(inbh2.getWeight() + inbh2_change);
         //update weights for connections to output layer:
         h1out.updateWeight(h1out.getWeight() + h1out_change); //give it a boost in same direction as last time
         h2out.updateWeight(h2out.getWeight() + h2out_change);
         hbout.updateWeight(hbout.getWeight() + hbout_change);
         
         //set current weightChanges as previousChange for next iteration for all connections
         in1h1.setPrevChange(in1h1_change);
         in1h2.setPrevChange(in1h2_change);
         in2h1.setPrevChange(in2h1_change);
         in2h2.setPrevChange(in2h2_change);
         inbh1.setPrevChange(inbh1_change);
         inbh2.setPrevChange(inbh2_change);
         h1out.setPrevChange(h1out_change);
         h2out.setPrevChange(h2out_change);
         hbout.setPrevChange(hbout_change);
               
       }
       // evaluate network's performance
       double globalError = 0.0;
       for (int i = 0; i < trainingData.size(); i++) {
         Double[] data = trainingData.get(i);
         double target = desiredVals[i];
         input1.setInputVal(data[0]);
         input2.setInputVal(data[1]);
         inputBias.setInputVal(1);
         hiddenBias.setInputVal(1);
         //hidden layer computes values
         hidden1.computeValue();
         hidden2.computeValue();
         //output computes value 
         outputValue = output.computeValue();
         //compute error 
         error = target - outputValue;
         globalError += error*error;
       }
       if (cycle == trainingCycles-1) {
          System.out.printf("Global Error after %d cycles is %f\n", trainingCycles, globalError);
       }
     } 
     //print final global error, and go through training data and print out outputs to see how the training fared
     for (int i = 0; i < trainingData.size(); i++) {
         Double[] data = trainingData.get(i);
         double target = desiredVals[i];
         input1.setInputVal(data[0]);
         input2.setInputVal(data[1]);
         inputBias.setInputVal(1);
         hiddenBias.setInputVal(1);
         hidden1.computeValue();
         hidden2.computeValue();
         outputValue = output.computeValue();
         System.out.printf("Output for %.1f XOR %.1f is %f and should be %.1f\n", data[0], data[1], outputValue, target);
     }
     System.out.println();
     System.out.println();  
  }
}
}
  
