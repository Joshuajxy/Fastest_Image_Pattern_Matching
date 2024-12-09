import os
import sys
import cv2
import numpy as np
import ctypes
from ctypes import WinError

# 添加DLL搜索路径
dll_dir = r"C:\Users\joshua.yang\code\tmp\Fastest_Image_Pattern_Matching\py_matcher\out\build\x64-Debug\lib"
os.environ['PATH'] = dll_dir + os.pathsep + os.environ['PATH']

# 打印环境变量以验证
print("PATH:", os.environ['PATH'])

# 添加模块路径
sys.path.append(dll_dir)

# 检查DLL是否存在
opencv_dll = os.path.join(dll_dir, "opencv_world4100d.dll")
print(f"OpenCV DLL exists: {os.path.exists(opencv_dll)}")

def verify_environment():
    print("\n=== Environment Verification ===")
    print("Current directory:", os.getcwd())
    
    pyd_path = os.path.join(dll_dir, "cvmatcher.cp311-win_amd64.pyd")
    print("PYD file exists:", os.path.exists(pyd_path))
    print("PYD file path:", pyd_path)
    
    try:
        print("\nTrying to load DLL directly...")
        dll = ctypes.CDLL(pyd_path)
        print("DLL loaded successfully")
    except Exception as e:
        print("DLL load error:", e)

def test_basic_match():
    try:
        print("\n=== Testing Basic Match ===")
        print("Importing cvmatcher...")
        import cvmatcher
        print("cvmatcher imported successfully")
        
        print("Creating matcher instance...")
        matcher = cvmatcher.TemplateMatcher()
        
        print("Loading images...")
        template = cv2.imread("template_image.jpg", cv2.IMREAD_GRAYSCALE)
        if template is None:
            print("Error: Could not load template image")
            return
        print(f"Template size: {template.shape}")
            
        source = cv2.imread("source_image.jpg", cv2.IMREAD_GRAYSCALE)
        if source is None:
            print("Error: Could not load source image")
            return
        print(f"Source size: {source.shape}")

        print("Setting parameters...")
        matcher.set_template(template)
        matcher.set_max_positions(5)
        matcher.set_source(source)
        matcher.set_score(0.5)
        matcher.set_tolerance_angle(180.0)
        
        print("Executing matching...")
        results = matcher.match()
        print(f"Found {len(results)} matches")
        
        output = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        for result in results:
            try:
                print("\nRaw result:", result)
                
                center_x = result.pt_center[0]
                center_y = result.pt_center[1]
                
                cv2.circle(output, (int(center_x), int(center_y)), 5, (0,255,0), -1)
                print(f"Match at ({center_x:.1f}, {center_y:.1f}) "
                      f"score: {result.match_score:.3f}, angle: {result.matched_angle:.1f}")
                      
                pts = np.array([
                    [int(result.pt_lt[0]), int(result.pt_lt[1])],
                    [int(result.pt_rt[0]), int(result.pt_rt[1])],
                    [int(result.pt_rb[0]), int(result.pt_rb[1])],
                    [int(result.pt_lb[0]), int(result.pt_lb[1])]
                ], np.int32)
                cv2.polylines(output, [pts], True, (0,255,0), 2)
                
            except Exception as e:
                print(f"Error processing result: {str(e)}")
                print(f"Result data: {dir(result)}")
        
        cv2.imshow("Results", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error during matching: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_environment()
    test_basic_match()