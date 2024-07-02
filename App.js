import React, { useEffect, useState } from 'react';
import { StyleSheet, Text, View, Dimensions } from 'react-native';
import { LineChart } from 'react-native-chart-kit';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import { registerRootComponent } from 'expo';

const { width } = Dimensions.get('window');

const App = () => {
  const [trainAcc, setTrainAcc] = useState([]);
  const [testAcc, setTestAcc] = useState([]);
  const [epochs, setEpochs] = useState([]);

  useEffect(() => {
    const loadAndTrainModel = async () => {
      await tf.ready();

      const model = tf.sequential();
      model.add(tf.layers.dense({ units: 32, activation: 'relu', inputShape: [784] }));
      model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
      model.compile({ optimizer: 'sgd', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

      // Dummy data for demonstration
      const parsedTrainAcc = [0.1, 0.2, 0.3, 0.4];
      const parsedTestAcc = [0.1, 0.15, 0.25, 0.35];
      const parsedEpochs = [1, 2, 3, 4];

      setTrainAcc(parsedTrainAcc);
      setTestAcc(parsedTestAcc);
      setEpochs(parsedEpochs);
    };

    loadAndTrainModel();
  }, []);

  return (
    <View style={styles.container}>
      <Text style={styles.title}>MNIST Training and Testing Accuracy</Text>
      <LineChart
        data={{
          labels: epochs.map(e => e.toString()),
          datasets: [
            { data: trainAcc, color: (opacity = 1) => `rgba(255, 0, 0, ${opacity})`, label: 'Training Accuracy' },
            { data: testAcc, color: (opacity = 1) => `rgba(0, 0, 255, ${opacity})`, label: 'Testing Accuracy' }
          ]
        }}
        width={width - 16}
        height={220}
        chartConfig={{
          backgroundColor: '#e26a00',
          backgroundGradientFrom: '#fb8c00',
          backgroundGradientTo: '#ffa726',
          decimalPlaces: 2,
          color: (opacity = 1) => `rgba(255, 255, 255, ${opacity})`,
          labelColor: (opacity = 1) => `rgba(255, 255, 255, ${opacity})`,
          style: { borderRadius: 16 },
          propsForDots: { r: '6', strokeWidth: '2', stroke: '#ffa726' }
        }}
        style={{ marginVertical: 8, borderRadius: 16 }}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#fff' },
  title: { fontSize: 24, marginBottom: 16 },
});

export default App;

// Register the main component
registerRootComponent(App);
