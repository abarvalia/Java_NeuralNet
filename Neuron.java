import java.util.ArrayList;
import java.util.Random;

public class Neuron {
  
  //instance variables
  private double value;
  private String type;
  private double product;
  private ArrayList<Connection> connections = new ArrayList<Connection>();
  
  //for the input neurons, use this constructor
  public Neuron(String type) {
    this.value = 0.0;
    this.type = type;
    this.product = 0.0;
  }
  
  public String getType() {
    if (this.type.equals("Input")) {
      return "Input";
    }
    else if (this.type.equals("Hidden")) {
      return "Hidden";
    }
    else if (this.type.equals("Bias")) {
      return "Bias";
    }
    else {
      return "Output";
    }
  }
  
  public double getValue() {
    return this.value;
  }
  
  public void setValue(double value) {
    this.value = value;
  }
  
  public void setConnectionsArray(Connection connection) {
    connections.add(connection);
  }
  
  public double sigmoid(double x) {
    return (1/( 1 + Math.pow(Math.E,(-1*x))));
  }
  
  public double computeValue() {
    if (this.getType().equals("Output") || this.getType().equals("Hidden")) {
      this.value = 0;
      for (Connection connection : connections) {
        this.value = this.value + connection.getTriggerVal();
      }
      this.value = this.sigmoid(this.value);
      return this.value;
    }
    else {
      return 0.0;
    }
  }
  
  public double setRandomWeight() {
    double weight;
    Random r = new Random(); 
    weight = -0.1 + r.nextDouble() * 0.2;
    return weight;
  }
  
  
  public void setInputVal(double input) {
    this.value = input;
  }

}