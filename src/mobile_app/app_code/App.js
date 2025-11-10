import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import SetupScreen from './screens/SetupScreen';
import RecordingScreen from './screens/RecordingScreen';
import InferenceScreen from './screens/InferenceScreen';

const Stack = createStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Setup">
        <Stack.Screen 
          name="Setup" 
          component={SetupScreen}
          options={{ title: 'Smooth Pursuit Setup' }}
        />
        <Stack.Screen 
          name="Recording" 
          component={RecordingScreen}
          options={{ title: 'Data Collection', headerShown: false }}
        />
        <Stack.Screen 
          name="Inference" 
          component={InferenceScreen}
          options={{ title: 'Smooth Pursuit Test', headerShown: false }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

