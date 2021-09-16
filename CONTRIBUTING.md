# Contributing

This `Contributing` file came mostly from the react native template. I don't care much for linting rules. And don't anyone ever add `Prettier` to this.
As long as you comment your code well, it functions well, and performance doesn't drop, everything's good.

**DO NOT SUBMIT PULL REQUESTS TO CONVERT THE iOS PART OF THIS PROJECT TO SWIFT. THIS PROJECT WILL ALWAYS BE OBJECTIVE-C/C FIRST**

## Development workflow

To get started with the project, run `yarn` in the root directory to install the required dependencies for each package:

```sh
yarn
```

> While it's possible to use [`npm`](https://github.com/npm/cli), the tooling is built around [`yarn`](https://classic.yarnpkg.com/), so you'll have an easier time if you use `yarn` for development.

While developing, you can run the [example app](/example/) to test your changes. Any changes you make in your library's JavaScript code will be reflected in the example app without a rebuild. If you change any native code, then you'll need to rebuild the example app.

To start the packager:

```sh
yarn example start
```

To run the example app on Android:

```sh
yarn example android
```

To run the example app on iOS:

```sh
yarn example ios
```

Make sure your code passes TypeScript and ESLint. Run the following to verify:

```sh
yarn typescript
yarn lint
```

To fix formatting errors, run the following:

```sh
yarn lint --fix
```

Remember to add tests for your change if possible. Run the unit tests by:

```sh
yarn test
```

To edit the Objective-C files, open `example/ios/TensorflowLiteExample.xcworkspace` in XCode and find the source files at `Pods > Development Pods > react-native-tensorflow-lite`.

To edit the Kotlin files, open `example/android` in Android studio and find the source files at `reactnativetensorflowlite` under `Android`.

### Commit message convention

Screw commit conventions. Just make sure to write something if you feel like you did something significant.

### Linting and tests

We use [TypeScript](https://www.typescriptlang.org/) for type checking.

That's it. ESLint is mostly just for suggestions. *NO PRETTIER ALLOWED IN MY REPOS*

### Scripts

The `package.json` file contains various scripts for common tasks:

- `yarn bootstrap`: setup project by installing all dependencies and pods.
- `yarn typescript`: type-check files with TypeScript.
- `yarn lint`: lint files with ESLint.
- `yarn test`: run unit tests with Jest.
- `yarn example start`: start the Metro server for the example app.
- `yarn example android`: run the example app on Android.
- `yarn example ios`: run the example app on iOS.

### Sending a pull request

> **Working on your first pull request?** You can learn how from this _free_ series: [How to Contribute to an Open Source Project on GitHub](https://egghead.io/series/how-to-contribute-to-an-open-source-project-on-github).

When you're sending a pull request:

- Prefer small pull requests focused on one change.
- Verify that linters and tests are passing.
- Review the documentation to make sure it looks good.
- Follow the pull request template when opening a pull request.
- For pull requests that change the API or implementation, discuss with maintainers first by opening an issue.

## Code of Conduct

Just write good code, keep it simple. Don't intentionally break existing stuff without discussion. Everyone's happy.