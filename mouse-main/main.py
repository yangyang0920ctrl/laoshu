import os
import cv2
import torch
import streamlit as st
from PIL import Image
import datetime
import torchvision.models as m
from torchvision import transforms
import tempfile

# Constants
LABELS = {
    "0": "Convusion",
    "1": "anxiety",
    "2": "body_twitching",
    "3": "exploratory_moving",
    "4": "extend_limbs",
    "5": "head_shaking",
    "6": "moderate_dysphea",
    "7": "scratching",
    "8": "severe_dysphea",
    "9": "washing_face"
}


def create_model(model, num, weights):
    net = m.efficientnet_b0(weights=m.EfficientNet_B0_Weights.DEFAULT if weights else False, progress=True)
    tmp = list(net.classifier)[-1].in_features
    net.classifier = torch.nn.Linear(tmp, num, bias=True)
    return net


def data_trans(train_mean=[0.485, 0.456, 0.406], train_std=[0.229, 0.224, 0.225]):
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomRotation(90),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std)
    ])

    return train_transform, test_transform


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.sidebar.info(f"Using device: {device}")
    return device


def format_time(seconds):
    return str(datetime.timedelta(seconds=seconds)).split(".")[0]


def process_video(video_file, show_results, progress_bar, status_text):
    device = get_device()
    _, data_transform = data_trans()

    # Create model
    net = create_model(model='b0', num=len(LABELS), weights=False)
    net.load_state_dict(torch.load('./runs/weights/best.pth', map_location=device), strict=False)
    net.to(device)
    net.eval()

    # Create temp file for video
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_video.write(video_file.read())
    temp_video.close()

    # Video input
    cap = cv2.VideoCapture(temp_video.name)
    if not cap.isOpened():
        status_text.error("无法打开视频文件")
        return

    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare results container
    results = []
    current_second = -1
    last_class = None
    start_time = 0
    end_time = 0

    frame_count = 0
    video_placeholder = st.empty()
    stop_button = st.sidebar.button("停止处理")

    # Create results display area in sidebar
    results_placeholder = st.sidebar.empty()
    if show_results:
        results_placeholder.markdown("### 分类结果")

    with torch.no_grad():
        while cap.isOpened():
            if stop_button:
                status_text.warning("处理已停止")
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            progress = int((frame_count / total_frames) * 100)
            progress_bar.progress(progress)
            status_text.text(f"处理中... {progress}% 完成")

            current_time = frame_count / fps
            second = int(current_time)

            # Convert frame to PIL Image and preprocess
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = data_transform(pil_img)
            img = torch.unsqueeze(img, dim=0)

            # Predict
            output = net(img.to(device))
            output = torch.softmax(output, dim=1)
            p, index = torch.topk(output, k=3)
            current_class = LABELS[str(index.to("cpu").numpy()[0][0])]

            # Display top 3 probabilities in the frame
            text_y = 30
            for i in range(3):
                class_idx = index.to("cpu").numpy()[0][i]
                class_name = LABELS[str(class_idx)]
                prob = p.to("cpu").numpy()[0][i]
                text = f'{class_name}: {prob:.4f}'
                cv2.putText(frame, text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                text_y += 30

            # Check results every second
            if second != current_second:
                current_second = second

                if current_class != last_class:
                    if last_class is not None and show_results:
                        time_range = f"{format_time(start_time)}-{format_time(end_time)}" if start_time != end_time else format_time(start_time)
                        if not results or results[-1] != f"{time_range}: {last_class}":
                            results.append(f"{time_range}: {last_class}")
                    start_time = second
                    last_class = current_class
                end_time = second

                # Update results display only when second changes
                if show_results and results:
                    results_placeholder.markdown("### 分类结果\n" + "\n".join(results))

            # Display the frame
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_image, channels="RGB")

    # Add the last time range if needed
    if last_class is not None and show_results:
        time_range = f"{format_time(start_time)}-{format_time(end_time)}" if start_time != end_time else format_time(start_time)
        if not results or results[-1] != f"{time_range}: {last_class}":
            results.append(f"{time_range}: {last_class}")
        results_placeholder.markdown("### 分类结果\n" + "\n".join(results))

    # Release resources
    cap.release()
    os.unlink(temp_video.name)


def main():
    st.set_page_config(
        page_title="智能视频分类工具",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("智能视频分类工具")

    # Sidebar for controls
    with st.sidebar:
        st.header("视频文件选择")
        video_file = st.file_uploader(
            "选择视频文件",
            type=["mp4", "avi", "mov"],
            label_visibility="collapsed"
        )

        st.header("输出选项")
        show_results = st.checkbox("显示分类结果", value=True)

        if st.button("开始分类"):
            if not video_file:
                st.error("请先选择视频文件")
            else:
                st.session_state.processing = True

    # Main content area
    if 'processing' in st.session_state and st.session_state.processing:
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            process_video(
                video_file,
                show_results,
                progress_bar,
                status_text
            )

            status_text.success("处理完成！")
            st.balloons()

        except Exception as e:
            status_text.error(f"处理过程中发生错误: {str(e)}")

        finally:
            st.session_state.processing = False

    # Display video info if selected
    if video_file and 'processing' not in st.session_state:
        st.subheader("视频信息")
        st.text(f"文件名: {video_file.name}")
        st.text(f"文件大小: {video_file.size / (1024 * 1024):.2f} MB")

        # Display first frame as preview
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_video.write(video_file.read())
        temp_video.close()

        cap = cv2.VideoCapture(temp_video.name)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(rgb_image, caption="视频预览 (第一帧)", channels="RGB")

        cap.release()
        os.unlink(temp_video.name)

        # Reset file pointer
        video_file.seek(0)


if __name__ == '__main__':
    main()
