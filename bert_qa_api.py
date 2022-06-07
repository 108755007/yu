import os.path
from fastapi import FastAPI
from transformers import AutoModelForQuestionAnswering,AutoTokenizer,pipeline
from db import DBhelper
import pandas as pd
import numpy as np
import re
import uvicorn
description = """
# BERT(PERT)
---模型來源:https://github.com/ymcui/PERT
"""
tags_metadata = [
    {
        "name": "禾多移動QA",
        "description": "可以提問禾多相關問題.",
    }
    ,
{
        "name": "資料庫客服回答",
        "description": "會回答得很爛且很慢,訊息雜亂/訊息量太多",
    },
]
app = FastAPI(title="Bert", description=description, openapi_tags=tags_metadata)


## jieba config
model = AutoModelForQuestionAnswering.from_pretrained('hfl/chinese-pert-base-mrc')
tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-pert-base-mrc')
QA = pipeline('question-answering', model=model, tokenizer=tokenizer)
def filter_str(text, pattern = "[https]{4,5}:\/\/[0-9a-zA-Z.\/].*"):
    text_clean = ''.join(re.split(pattern, text))
    #text_clean = ''.join(re.split(r'[\W]*', text_clean))
    text_clean = ''.join(re.split(r'[a-zA-Z]*', text_clean))
    text_clean = ''.join(re.split(r'\n', text_clean))
    text_clean = ''.join(re.split(r'\s', text_clean))
    return text_clean
def feach_message():
    # query1 = f"""
    #         SELECT message  FROM slack_line_customer_service_history line WHERE source NOT IN ('send_bot','alert_bot')
    #          """
    # message_1 = DBhelper('api02').ExecuteSelect(query1)

    query2 = f"""
            SELECT message  FROM slack_ig_customer_service_history line WHERE source NOT IN ('send_bot','alert_bot')
                """
    #message_2 = DBhelper('api02').ExecuteSelect(query2)

    query3 = f"""
            SELECT message  FROM slack_fb_customer_service_history line WHERE source NOT IN ('send_bot','alert_bot')
           """

    message_3 = DBhelper('api02').ExecuteSelect(query3)

    message = pd.DataFrame(message_3, columns=['message'])

    message = message.apply(lambda x: filter_str(x['message']), axis=1)

    content = str(list(set(message))[1:]).replace("'", '')

    return content
content = """
        禾多移動致力於AI推播服務，2017年開始串連電商及百大媒體品牌，即時分析用戶瀏覽行為，並對訂閱媒體推播的用戶打出精準廣告或推薦文章，成功為媒體及電商打開更大的獲客渠道，簡單來說，用戶收到的媒體推播，幾乎是來自禾多移動的技術，此為打造了四年的「流量池計畫」，截至2021年七月，禾多移動的流量池計畫累積擁有4,500萬訂閱用戶、八百大媒體品牌，三年內躍升成為台灣最大的媒體流量池。即便疫情衝擊實體經濟，電商品牌因受惠於禾多移動打造的「數位店長」服務，在線上仍維持高曝光及成交率，數位店長旨在提升消費者體驗，在沒有搜集第三方Cookies的情況下，仍能幫助電商業者找到消費者想要的產品，產品包含OnPage推播、關鍵字搜尋等，服務店家的轉單率皆提升了10%以上，其中食品保健商品收益更提升了23%。禾多經歷八年的時空磨礪，與第一線業者交手多年，深知品牌、企業要做到數位轉型的不容易，因此打造好用、好懂的AI產品，幫助品牌精準分析並預測消費者喜好，逐步實現「需求者和供應者能順利相遇」的願景。關於禾多移動創辦人林志堯：畢業於美國亞利桑那州立大學資工相關科系，曾任職台積電半導體研發中心，更與台積電夥伴成立了被稱為小聯發科的「晨星半導體」，專注於晶片研發，全球的市占率曾一度超過六成。林志堯不斷創新，勢要將台灣軟體優勢推向國際，證明台灣不只是硬體和代工國，更是智慧與軟體開發國家，因此與團隊接續成立台灣第一家APP廣告聯播網公司「酷手機」，後被智邦科技購併。2013年，林志堯再創立禾多移動，並立下「開發一種讓世界仰望台灣人的技術」的宣言，期許AI不只是技術和理念，要真正落實在產業應用
        禾多商品包含（烙郎,AviviD 廣告,數位店長,屏幕推播）。烙郎:讓你的直播瞬間擁入大量人潮。AviviD廣告:累積1300萬媒體訂閱用戶行為分析，做到真正的精準廣告。數位店長:商品標籤與商品推薦自動化，提高網站轉換率屏幕推播:打造自家流量池的最佳推播系統。禾多自我開發的『推薦引擎』是由多個以人工智慧技術為基礎而開發出來的自動化導流工具與方案所架構而成。大眾消費者經過長年累月的廣告轟炸，已經自我訓練出自動忽略廣告的能力，這也就是市面上廣告的效益不斷下滑。因此，禾多相信電商與媒體不但需要不斷對外曝光吸引更多的新用戶，更應該多加善用自己本身用戶的流量，多推出和用戶本身相關性高且有興趣的內容與產品。然而只有透過科技的運用才有辦法真正達到對全站用戶做分析與自動化執行的水準。『推薦引擎』所要提供給電商與媒體的就是精準的分析與自動化的客戶互動行為。今天，禾多的『推薦引擎』內的工具與方案服務台灣當地超過120家大型得媒體與知名的電商，每一秒都在替電商與媒體導入流量與營收。"""
content2=feach_message()
@app.get("/hodo_qa", tags=["禾多移動QA"])
def qa_(question: str=''):

    global composer
    if question=='':
        return {"message": "no sentence input", "data": ""}
    else:
        QA_dict = QA({'question': question, 'context':content}, topk=5, max_answer_len=99)
        a = []
        for i, j in enumerate(QA_dict):
            s, _, _, aa = j.values()
            if i == 0:
                a += [f"{aa},分數有{s}"]
            else:
                a += [f'{aa},分數:{s}']
        return {"最佳答案": a[0], "答案2": a[1], "答案3": a[2], "答案4": a[3], "答案5": a[4]}
@app.get("/hodo_qa2", tags=["資料庫客服回答"])
def qa_2(question: str=''):
    global composer
    if question=='':
        return {"message": "no sentence input", "data": ""}
    else:
        QA_dict = QA({'question': question, 'context':content2}, topk=5, max_answer_len=99)
        a = []
        for i, j in enumerate(QA_dict):
            s, _, _, aa = j.values()
            if i == 0:
                a += [f"{aa},分數有{s}"]
            else:
                a += [f'{aa},分數:{s}']
        return {"最佳答案": a[0], "答案2": a[1], "答案3": a[2], "答案4": a[3], "答案5": a[4]}

if __name__ == "__main__":
    uvicorn.run("main:app", host="192.168.0.150", port=8000, log_level="info")