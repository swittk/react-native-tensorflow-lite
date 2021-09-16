# react-native-tensorflow-lite

Tensorflow Lite for React Native
(Currently iOS only).

## Installation

```sh
npm install @switt/react-native-tensorflow-lite
```

## Current features
- Documented in Typescript. Seriously, just read the types from there.
- Allow selection of model simply by local file URL (no need to add to compile-time). Was designed to be used with Expo's Asset system (resolve model file URI from Asset.fromModule(...)) 
- Input type : Image(s)
  - Input 1 or multiple images to be run on a Tensorflow Lite model.
  - Allow selecting resize mode of the images (fit vs fill)
  - Allow specifying crop areas (especially useful when using a previous model to crop out a region of interest)
- iOS :: TensorflowLite with TF Ops enabled (version 2.6.0 - 0.0.1 nightly fails linking due to duplicated symbols. Will update if stable.)
  - simply `pod install` and the linker flags are added automatically.

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
