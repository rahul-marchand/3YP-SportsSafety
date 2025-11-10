import React, { useState, useEffect, useRef } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Dimensions } from 'react-native';
import { Camera, useCameraDevice } from 'react-native-vision-camera';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withTiming,
  Easing,
} from 'react-native-reanimated';
import io from 'socket.io-client';
import { BACKEND_URL, DEVICE_SPECS } from '../config';

const { width, height } = Dimensions.get('window');

export default function RecordingScreen({ route, navigation }) {
  const { distance } = route.params;
  const [isRecording, setIsRecording] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  const [status, setStatus] = useState('Ready');
  
  const device = useCameraDevice('front');
  const cameraRef = useRef(null);
  const socketRef = useRef(null);
  const frameIntervalRef = useRef(null);

  const dotX = useSharedValue(width / 2);

  useEffect(() => {
    (async () => {
      const permission = await Camera.requestCameraPermission();
      if (permission !== 'granted') {
        setStatus('Camera permission denied');
      }
    })();

    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, []);

  const startRecording = () => {
    const socket = io(`${BACKEND_URL}/ws/recording`, {
      transports: ['websocket'],
    });

    socketRef.current = socket;

    socket.on('connect', () => {
      socket.emit('message', JSON.stringify({
        type: 'init',
        distance,
        screen_width_cm: DEVICE_SPECS.screen.widthCm,
        screen_height_cm: DEVICE_SPECS.screen.heightCm,
        screen_width_px: DEVICE_SPECS.screen.widthPx,
        screen_height_px: DEVICE_SPECS.screen.heightPx,
        camera_x_px: DEVICE_SPECS.camera.positionPx.x,
        camera_y_px: DEVICE_SPECS.camera.positionPx.y,
      }));
    });

    socket.on('message', (data) => {
      const response = JSON.parse(data);
      if (response.status === 'saved') {
        setFrameCount(response.frame_count);
        setStatus(`Recorded ${response.frame_count} frames`);
      } else if (response.status === 'session_started') {
        setStatus('Recording...');
      }
    });

    dotX.value = withRepeat(
      withTiming(width - 50, {
        duration: 2000,
        easing: Easing.inOut(Easing.sin),
      }),
      -1,
      true
    );

    frameIntervalRef.current = setInterval(captureFrame, 100);
    setIsRecording(true);
  };

  const stopRecording = () => {
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
    }

    if (socketRef.current) {
      socketRef.current.emit('message', JSON.stringify({ type: 'stop' }));
      socketRef.current.close();
    }

    setIsRecording(false);
    setStatus(`Complete: ${frameCount} frames collected`);
  };

  const captureFrame = async () => {
    if (!cameraRef.current) return;

    try {
      const photo = await cameraRef.current.takePhoto({
        qualityPrioritization: 'speed',
        flash: 'off',
      });

      const base64 = await fetch(`file://${photo.path}`)
        .then(res => res.blob())
        .then(blob => {
          return new Promise((resolve) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result.split(',')[1]);
            reader.readAsDataURL(blob);
          });
        });

      socketRef.current.emit('message', JSON.stringify({
        type: 'frame',
        image: base64,
        dot_x: dotX.value,
        dot_y: height / 2,
      }));
    } catch (error) {
      console.error('Capture error:', error);
    }
  };

  const animatedDotStyle = useAnimatedStyle(() => ({
    transform: [{ translateX: dotX.value - 25 }],
  }));

  if (!device) {
    return (
      <View style={styles.container}>
        <Text>No camera device found</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Camera
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        photo={true}
      />

      <Animated.View style={[styles.dot, animatedDotStyle]} />

      <View style={styles.overlay}>
        <Text style={styles.statusText}>{status}</Text>
        
        <TouchableOpacity
          style={[styles.button, isRecording ? styles.stopButton : styles.startButton]}
          onPress={isRecording ? stopRecording : startRecording}
        >
          <Text style={styles.buttonText}>
            {isRecording ? 'STOP' : 'START RECORDING'}
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.backButton}
          onPress={() => navigation.goBack()}
        >
          <Text style={styles.buttonText}>BACK</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  dot: {
    position: 'absolute',
    top: height / 2 - 25,
    left: 0,
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: '#FF3B30',
  },
  overlay: {
    position: 'absolute',
    bottom: 40,
    left: 20,
    right: 20,
  },
  statusText: {
    color: '#fff',
    fontSize: 16,
    marginBottom: 20,
    textAlign: 'center',
  },
  button: {
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
    marginBottom: 12,
  },
  startButton: {
    backgroundColor: '#007AFF',
  },
  stopButton: {
    backgroundColor: '#FF3B30',
  },
  backButton: {
    backgroundColor: '#666',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  buttonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
});

