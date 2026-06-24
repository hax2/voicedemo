import os
import traceback
import sys

def test():
    try:
        from seed_vc_wrapper import get_wrapper
        vc_wrapper = get_wrapper()
        vc_wrapper.load()
        
        print("Model loaded. Testing convert_voice...")
        vc_wrapper.convert_voice(
            source_audio_path="test_edge.wav",
            reference_audio_path="test_edge.wav",
            diffusion_steps=25,
            length_adjust=1.0,
            inference_cfg_rate=0.7,
            use_cache=False
        )
    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    test()
