# react-native-tensorflow-lite

Tensorflow Lite for React Native
(Currently iOS only).

## Installation

```sh
npm install @switt/react-native-tensorflow-lite
```

## Usage

```ts
import TensorflowLite from "@switt/react-native-tensorflow-lite";

// ...
const imageUris : string[] = ['local_path_to_image_1', 'local_path_to_image_2']

const modelAsset = Asset.fromModule(require('./face_landmark.tflite'));
await modelAsset.downloadAsync();

// Operation is run in batches of files. This is to minimize the number of react native bridge calls.
const results = await TensorflowLite.runModelWithFiles({
  model: modelAsset.localUri!,
  files: imageUris
});
this.setState({ results });
```

## Contributing

See the [contributing guide](CONTRIBUTING.md) to learn how to contribute to the repository and the development workflow.

## License

MIT
