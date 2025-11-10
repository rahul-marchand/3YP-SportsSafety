import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet } from 'react-native';

export default function SetupScreen({ navigation }) {
  const [distance, setDistance] = useState('30');

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Smooth Pursuit Eye Tracking</Text>
      
      <Text style={styles.label}>Viewing Distance (cm):</Text>
      <TextInput
        style={styles.input}
        value={distance}
        onChangeText={setDistance}
        keyboardType="numeric"
        placeholder="30"
      />

      <Text style={styles.instructions}>
        Hold phone in landscape mode at the specified distance from your eyes.
        Keep your head still and follow the dot with your eyes only.
      </Text>

      <TouchableOpacity
        style={[styles.button, styles.recordButton]}
        onPress={() => navigation.navigate('Recording', { distance: parseInt(distance) })}
      >
        <Text style={styles.buttonText}>START RECORDING</Text>
      </TouchableOpacity>

      <TouchableOpacity
        style={[styles.button, styles.inferenceButton]}
        onPress={() => navigation.navigate('Inference', { distance: parseInt(distance) })}
      >
        <Text style={styles.buttonText}>START TEST</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#fff',
    justifyContent: 'center',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 30,
    textAlign: 'center',
  },
  label: {
    fontSize: 16,
    marginBottom: 10,
    fontWeight: '600',
  },
  input: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 12,
    fontSize: 18,
    marginBottom: 20,
  },
  instructions: {
    fontSize: 14,
    color: '#666',
    marginBottom: 30,
    lineHeight: 20,
  },
  button: {
    padding: 16,
    borderRadius: 8,
    marginBottom: 12,
    alignItems: 'center',
  },
  recordButton: {
    backgroundColor: '#007AFF',
  },
  inferenceButton: {
    backgroundColor: '#34C759',
  },
  buttonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
});

