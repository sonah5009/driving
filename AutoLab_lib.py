
import base64
import cv2
import warnings
import sys

# 암호화된 코드
ENCRYPTED = b'aW1wb3J0IGN2MgppbXBvcnQgd2FybmluZ3MKaW1wb3J0IHN5cwpzeXMuc3Rkb3V0LnJlY29kaW5nID0gJ3V0Zi04JwoKbWVzc2FnZSA9ICIiIgogICAgXyAgICAgX19fICAgICAgICAgX19fXyAgICAgICAgICAgX19fXyAgCiAgIC8gXCAgIHxfIF98ICAgICAgIC8gX19ffCAgIF9fXyAgIC8gX19ffCAKICAvIF8gXCAgIHwgfCAgX19fX18gXF9fXyBcICAvIF8gXCB8IHwgICAgIAogLyBfX18gXCAgfCB8IHxfX19fX3wgX19fKSB8fCAoXykgfHwgfF9fXyAgCi9fLyAgIFxfXHxfX198ICAgICAgIHxfX19fLyAgXF9fXy8gIFxfX19ffCAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogX19fXyAgICAgICAgICAgXyAgICAgICAgICAgXyAgICAgICAgICAgICAgICAKfCAgXyBcICAgXyBfXyAgKF8pX18gICBfXyAoXykgIF8gX18gICAgX18gXyAKfCB8IHwgfCB8ICdfX3wgfCB8XCBcIC8gLyB8IHwgfCAnXyBcICAvIF9gIHwKfCB8X3wgfCB8IHwgICAgfCB8IFwgViAvICB8IHwgfCB8IHwgfHwgKF98IHwKfF9fX18vICB8X3wgICAgfF98ICBcXy8gICB8X3wgfF98IHxffCBcX18sIHwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgfF9fXy8gCiAgX19fXyAgIF8gICAgICAgICAgICAgICAgICAgIAogLyBfX198IHwgfCAgIF9fIF8gIF9fXyAgX19fIAp8IHwgICAgIHwgfCAgLyBfYCB8LyBfX3wvIF9ffAp8IHxfX18gIHwgfCB8IChffCB8XF9fIFxfXyBcIAogXF9fX198IHxffCAgXF9fLF98fF9fXy98X19fLwogICAgICAgICAgICAgICAgICAgICAgICAgICAKCiIiIgoKZGVmIGluaXQoKToKICAgICMgT3BlbkNWIOqyveqzoCDrrLTsi5wKICAgIGN2Mi5zZXRMb2dMZXZlbCgwKQogICAgCiAgICAjIGtleWJvYXJkIOuqqOuTiOydmCBVc2VyV2FybmluZyDrrLTsi5wKICAgIHdhcm5pbmdzLmZpbHRlcndhcm5pbmdzKCJpZ25vcmUiLCBjYXRlZ29yeT1Vc2VyV2FybmluZywgbW9kdWxlPSJrZXlib2FyZCIpCiAgICAKICAgIHByaW50KG1lc3NhZ2UpCiAgICBwcmludCgiQUktU29DIOyekOycqOyjvO2WiSDqtZDsnKHqs7zsoJUiKQogICAgcHJpbnQoJ1N1bmdreXVua3dhbiBVbml2ZXJzaXR5IEF1dG9tYXRpb24gTGFiLicpCiAgICBwcmludCgnJykKICAgIHByaW50KCctLS0tLS0tLS0tLS0tLS0tLS1BdXRob3JzLS0tLS0tLS0tLS0tLS0tLS0tJykKICAgIHByaW50KCdHeXVoeWVvbiBId2FuZyA8cmJndXM3MDgwQGcuc2trdS5lZHU+JykKICAgIHByaW50KCdIb2JpbiBPaCA8aG9iaW4wNjc2QGRhdW0ubmV0PicpICAgIAogICAgcHJpbnQoJ01pbmt3YW4gQ2hvaSA8YXJib25nOTdAbmF2ZXIuY29tPicpICAgIAogICAgcHJpbnQoJ0h5ZW9uamluIFNpbSA8bnVmeHdtc0BuYXZlci5jb20+JykgICAgCiAgICBwcmludCgnSHllb25nS2V1biBIb25nIDx3aGFpaG9uZ0BnLnNra3UuZWR1PicpCiAgICBwcmludCgnRXVuSG8gS2ltIDxkbXNnaGRtc3RqQGcuc2trdS5lZHU+ICcpCiAgICBwcmludCgnWWVvbmdnd2FuZyBDaG9pIDxkdWRyaGtkNzgxMUBnLnNra3UuZWR1PicpCiAgICBwcmludCgnWW91bmdIb29uIFN1aCA8ZHVkZ25zMDQwN0BnLnNra3UuZWR1PicgKQogICAgcHJpbnQoJ0h5b2ppbiBQYXJrIDxwaHlvamluMDUxMUBnLnNra3UuZWR1PicgKQogICAgcHJpbnQoJ1lvdW5nIFNvbyBEbyA8Y29rMjUyOUBuYXZlci5jb20+JykKICAgIHByaW50KCdTdW5nIEJoaW4gT2ggPG9zYjgyNTJAZ21haWwuY29tPicpICAgICAgICAKICAgIHByaW50KCdIeWVva0p1biBDaG9pIDxtaWNrOTVAbmF2ZXIuY29tPicpCiAgICBwcmludCgnU3VuZ2tldW4gQ2hhIDxjaGFzazE3QGcuc2trdS5lZWR1PicpCiAgICBwcmludCgnU2VvbmctSHllb24gTGltIDxqb29uaG9mbG93QGcuc2trdS5lZHU+JykKICAgIHByaW50KCdCZW9tZ2kgU3VoIDxmb3Jmb3J0dW5hQHNra3UuZWR1PicpCiAgICBwcmludCgnWW91bmdtaW4gS2ltIDxiaWdibmd0ZW5AZy5za2t1LmVkdT4nKQogICAgcHJpbnQoJy0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLScpCiAgICBwcmludCgi7ZmY6rK97ISk7KCVIOuhnOuUqeykkS4uLiIp'

def init():
    """시스템 초기화 함수"""
    # OpenCV 로그 레벨 설정
    cv2.setLogLevel(0)
    
    # 키보드 경고 무시
    warnings.filterwarnings("ignore", category=UserWarning, module="keyboard")
    
    # 프로젝트 정보 출력
    print("AI-SoC 자율주행 시스템")
    print('Sungkyunkwan University Automation Lab')
    print('')
    print('-------------------Authors-------------------')
    print('Gyuhyeon Hwang <rbgus7080@g.skku.edu>')
    print('Hobin Oh <hobin0676@daum.net>')    
    print('Minkwan Choi <arbong97@naver.com>')    
    print('Hyeonjin Sim <nufxwms@naver.com>')    
    print('HyeongKeun Hong <whaihong@g.skku.edu>')
    print('EunHo Kim <dmsghdmstj@g.skku.edu> ')
    print('Yeonggwang Choi <dudrhkd7811@g.skku.edu>')
    print('YoungHoon Suh <dudgns0407@g.skku.edu> ')
    print('HyoJin Park <phyoJin0511@g.skku.edu> ')
    print('Young Soo Do <cok2529@naver.com>')
    print('Sung Bin Oh <osb8252@gmail.com>')        
    print('HyeokJun Choi <mick95@naver.com>')
    print('Sungkeun Cha <chask17@g.skku.edu>')
    print('Seong-Hyeon Lim <joonhoflow@g.skku.edu>')
    print('Beomgi Suh <forfortuna@skku.edu>')
    print('Youngmin Kim <bigbangten@g.skku.edu>')
    print('--------------------------------------------')
    print("시스템을 초기화합니다...")

# 복호화 및 실행
exec(base64.b64decode(ENCRYPTED).decode('utf-8'))
