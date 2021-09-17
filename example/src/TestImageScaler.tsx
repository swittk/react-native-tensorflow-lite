import * as React from 'react';

import { StyleSheet, View, Text, Alert, Button, Image, ScrollView, StyleProp, ViewStyle } from 'react-native';
import TensorflowLite, { SKTFLiteSingleTensorResult, SKTFLiteTensorResult } from '@switt/react-native-tensorflow-lite';
import { Asset } from 'expo-asset';
import * as ImagePicker from 'expo-image-picker';
import { base64RawToDataURI } from './utils';


export const TestImageScaler = React.memo((props: {
  style?: StyleProp<ViewStyle>,
  imageUri: string,
  faceLandmarkResults?: SKTFLiteTensorResult[]
}) => {
  const {
    style,
    imageUri,
    faceLandmarkResults
  } = props;

  const [transformedImage, setTransformedImage] = React.useState<string>()
  return <View style={style}>
    {
      faceLandmarkResults ?
        <RenderFaceBox imageUri={imageUri} face={faceLandmarkResults[0][0]} />
        :
        <Image source={{ uri: imageUri, width: 250, height: 250 }} resizeMode='contain' />
    }
    <Button title='TRANSFORM!' onPress={async () => {
      const result = await TensorflowLite.tensorImageTest({
        file: imageUri,
        // size: { width: 100, height: 100 },
        relativeCrops: [0, 0.0, 0.5, 0.5],
        backgroundColor: '#000000'
      });
      setTransformedImage(base64RawToDataURI(result));
    }} />
    <Button title='TRANSFORM TO EYE' onPress={async () => {
      if (!faceLandmarkResults) {
        Alert.alert('no face landmarks yet');
        return;
      }
      const face = faceLandmarkResults[0];
      const isRightEye = false;
      const leftEyeV = [417, 419, 276, 340];
      const rightEyeV = [193, 196, 46, 111];
      const edgePoints = isRightEye ? rightEyeV : leftEyeV;
      const edgeCoords = edgePoints.map((pIdx) => {
        const startIdx = pIdx * 3;
        const x = face[0].data[startIdx];
        const y = face[0].data[startIdx + 1];
        const z = face[0].data[startIdx + 2];
        return { x, y, z }
      });
      const xcoords = edgeCoords.map((v) => v.x);
      const ycoords = edgeCoords.map((v) => v.y);
      const cropRight = Math.max(...xcoords);
      const cropLeft = Math.min(...xcoords);
      const cropTop = Math.min(...ycoords);
      const cropBottom = Math.max(...ycoords);
      // return { cropTop, cropLeft, cropRight, cropBottom };
      const crops = [cropLeft / 192.0, cropTop / 192.0, (cropRight - cropLeft) / 192.0, (cropBottom - cropTop) / 192.0] as [number, number, number, number];
      Alert.alert(`cropping to ${isRightEye ? 'right' : 'left'} eye crops`, JSON.stringify(crops));
      const result = await TensorflowLite.tensorImageTest({
        file: imageUri,
        size: { width: 100, height: 100 },
        relativeCrops: crops,
        backgroundColor: '#000000'
      });
      setTransformedImage(base64RawToDataURI(result));
    }} />
    <Text>Transformed Image Result</Text>
    <Image source={{ uri: transformedImage, width: 250, height: 250 }} resizeMode='contain' />
    <Button title='Transformed image info' onPress={async () => {
      if (!transformedImage) { Alert.alert('no image'); return; }
      const prom = new Promise<{ width: number, height: number }>((resolve, reject) => {
        Image.getSize(transformedImage, (w, h) => {
          resolve({ width: w, height: h });
        }, reject);
      });
      const dims = await prom;
      Alert.alert(`Image dims`, JSON.stringify(dims));
    }} />
  </View>
})

export const RenderFaceBox = React.memo((props: {
  face: SKTFLiteSingleTensorResult,
  inputTensorWidth?: number,
  inputTensorHeight?: number,
  style?: StyleProp<ViewStyle>,
  imageUri?: string
}) => {
  const {
    style,
    face,
    inputTensorWidth = 192,
    inputTensorHeight = 192,
    imageUri
  } = props;
  const [imSize, setImSize] = React.useState({ width: 250, height: 250 });
  React.useEffect(() => {
    if (!imageUri) return;
    (async () => {
      const prom = new Promise<{ width: number, height: number }>((resolve, reject) => {
        Image.getSize(imageUri, (w, h) => {
          resolve({ width: w, height: h });
        }, reject);
      });
      const size = await prom;
      setImSize(size);
    })();
  }, [imageUri]);
  // shape of FaceMesh output is expected to be (1,1,1,1404)
  // console.log('rendered', faces.length, 'faces');
  // const validArray = React.useMemo(() => {
  //   return arrayEqual(face.shape, [1, 1, 1, 1404]);
  // }, [face]);
  const data = face.data;
  const datalen = data.length;
  const landmarkCircles: JSX.Element[] = [];
  for (let i = 0; i < datalen; i += 3) {
    const x = data[i];
    const y = data[i + 1];
    // const z = data[i + 2];
    landmarkCircles.push(<View
      key={`landmark_${i}`}
      style={{
        width: 1.5, height: 1.5, borderRadius: 0.5,
        left: `${x / inputTensorWidth * 100}%`,
        top: `${y / inputTensorHeight * 100}%`,
        backgroundColor: 'blue',
        position: 'absolute'
      }}
    />);
  }

  return <View
    style={[{ width: 250, height: 250 }, style]}
  >
    <Image source={{ uri: imageUri }} style={{ aspectRatio: imSize.width / imSize.height, flex: 1 }} resizeMode='contain' />
    <View style={StyleSheet.absoluteFill} pointerEvents='none'>
      <View style={{ aspectRatio: imSize.width / imSize.height, flex: 1 }}>
        {landmarkCircles}
      </View>
    </View>
  </View>;
})

function arrayEqual(a1: any[], a2: any[]) {
  let i = a1.length;
  while (i--) {
    if (a1[i] !== a2[i]) return false;
  }
  return true
}