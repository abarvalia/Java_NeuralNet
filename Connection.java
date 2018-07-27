public class Connection {
  
  //instance variables
  private Neuron sourceNeuron;
  private Neuron targetNeuron;
  private double triggerVal;
  private double weight;
  private double prevChange;

  
  public Connection(Neuron sourceNeuron, Neuron targetNeuron, double weight) {
    this.sourceNeuron = sourceNeuron;
    this.targetNeuron = targetNeuron;
    this.weight = weight;
    this.prevChange = 0.0;
  }
  
  public double getTriggerVal() {
    return sourceNeuron.getValue() * this.weight;
  }
  
  public double getWeight() {
    return this.weight;
  }
  
  public void updateWeight(double newWeight) {
    this.weight = newWeight;
  }
  
  public double getPrevChange() {
    return this.prevChange;
  }
  
  public void setPrevChange(double change) {
    this.prevChange = change;
  }
  
}