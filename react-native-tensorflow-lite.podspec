require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))

Pod::Spec.new do |s|
  s.name         = "react-native-tensorflow-lite"
  s.version      = package["version"]
  s.summary      = package["description"]
  s.homepage     = package["homepage"]
  s.license      = package["license"]
  s.authors      = package["author"]

  s.platforms    = { :ios => "10.0" }
  s.source       = { :git => "https://github.com/swittk/react-native-tensorflow-lite.git", :tag => "#{s.version}" }

  s.source_files = "ios/**/*.{h,m,mm}"

  s.dependency "React-Core"
  s.dependency 'TensorFlowLiteObjC', '~> 2.6.0'
  s.dependency 'TensorFlowLiteObjC/Metal', '~> 2.6.0'
  s.dependency 'TensorFlowLiteSelectTfOps', '~> 2.6.0'
  s.user_target_xcconfig = { 'OTHER_LDFLAGS' => '-force_load $(SRCROOT)/Pods/TensorFlowLiteSelectTfOps/Frameworks/TensorFlowLiteSelectTfOps.framework/TensorFlowLiteSelectTfOps' }
end
