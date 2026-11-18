# @file exceptions.py
# @brief example のエラー: メッセージを stderr に出し、番号を付けて止まる
# @author Kohei Yamaguchi, Claude

##########################################################
'''
エラー番号
1. GPU がない。
2. エポックの数がおかしい。
3. FP32 の重みが、量子化したモデルに合わない。
4. CIFAR-10 のダウンロードが壊れている。
'''
##########################################################

import sys



class Error(SystemExit):
    '''
    #* example のエラー（メッセージ、詳しい内容、ヒント）を stderr に出し、その番号でプログラムを止める。
    #? main.py と utils/cifar10.py が投げる。番号の一覧はこのファイルの先頭にある。
    #! detail、hint、code はキーワード引数で渡す。
    '''
    
    def __init__(self, message: str, *, detail: str | None = None, hint: str | None = None, code: int) -> None:
        '''
        #* エラーを表示し、番号を終了コードとして持つ。
        '''
        
        lines = [f"[ERROR] {message}"] + ([detail] if detail else []) + ([f"Hint: {hint}"] if hint else [])
        print("\n".join(lines), file=sys.stderr)
        super().__init__(code)
        
        return None
