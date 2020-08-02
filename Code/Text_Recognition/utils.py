# -*- coding: utf-8 -*-

#!/usr/bin/python
# encoding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
import collections

#convert string to hex
def toHex(s):
    lst = []
    for ch in s:
        hv = hex(ord(ch)).replace('0x', '')
        if len(hv) == 1:
            hv = '0'+hv
        lst.append(hv)
    
    return reduce(lambda x,y:x+y, lst)

#convert hex repr to string
def toStr(s):
    return s and chr(atoi(s[:2], base=16)) + toStr(s[2:]) or ''

def convertFromChars(s):
    s = list(s)

    i = 0
    while True:
        try:
            char = s[i]
            if char.encode("hex") == "c3" or char.encode("hex") == "c5" or char.encode("hex") == "ce" or char.encode("hex") == "c2" or char.encode("hex") == "c6" or char.encode("hex") == "c9":
                s = s[0:i:] + s[i+1: :]
                next_char = s[i]
                if next_char.encode("hex") == "a0" and char.encode("hex") == "c5":
                    s[i] = "cc".decode("hex")
                elif next_char.encode("hex") == "b2" and char.encode("hex") == "c2":
                    s[i] = "cd".decode("hex")
                elif next_char.encode("hex") == "b9" and char.encode("hex") == "c6":
                    s[i] = "ce".decode("hex")
            elif char.encode("hex") == "db" or char.encode("hex") == "d0":
                s = s[0:i+1:] + s[i+2: :]
            elif char.encode("hex") == "e1" or char.encode("hex") == "e2" or char.encode("hex") == "e3":
                s = s[0:i:] + s[i+2: :]
                
                char = s[i]
                if char.encode("hex") == "a0":
                    s[i] = "c1".decode("hex")
                elif char.encode("hex") == "81":
                    s[i] = "c2".decode("hex")
                elif char.encode("hex") == "8f":
                    s[i] = "c3".decode("hex")
                elif char.encode("hex") == "aa":
                    s[i] = "c4".decode("hex")
                elif char.encode("hex") == "a2":
                    s[i] = "c5".decode("hex")
                elif char.encode("hex") == "ac":
                    s[i] = "c6".decode("hex")
                elif char.encode("hex") == "8a":
                    s[i] = "c7".decode("hex")
                elif char.encode("hex") == "8b":
                    s[i] = "c8".decode("hex")
                elif char.encode("hex") == "9c":
                    s[i] = "c9".decode("hex")
                elif char.encode("hex") == "9d":
                    s[i] = "ca".decode("hex")
                elif char.encode("hex") == "98":
                    s[i] = "cb".decode("hex")
        except IndexError:
            break
        i += 1
    s = "".join(s)
    return s

def convertToChars(s):
    s = list(s)

    i = 0
    while True:
        try:
            char = s[i]
            if char == "a0".decode("hex"):
                s[i] = u"à"
            elif char == "cc".decode("hex"):
                s[i] = u"Š"
            elif char == "91".decode("hex"):
                s[i] = u"ɑ"
            elif char == "db".decode("hex"):
                s[i] = u"۸"
            elif char == "d0".decode("hex"):
                s[i] = u"з"
            elif char == "b2".decode("hex"):
                s[i] = u"ò"
            elif char == "cd".decode("hex"):
                s[i] = u"²"
            elif char == "b9".decode("hex"):
                s[i] = u"ù"
            elif char == "ce".decode("hex"):
                s[i] = u"ƹ"
            elif char == "c1".decode("hex"):
                s[i] = u"Ṡ"
            elif char == "c2".decode("hex"):
                s[i] = u"▁"
            elif char == "c3".decode("hex"):
                s[i] = u"●"
            elif char == "c4".decode("hex"):
                s[i] = u"▪"
            elif char == "c5".decode("hex"):
                s[i] = u"•"
            elif char == "c6".decode("hex"):
                s[i] = u"€"
            elif char == "c7".decode("hex"):
                s[i] = u"《"
            elif char == "c8".decode("hex"):
                s[i] = u"》"
            elif char == "c9".decode("hex"):
                s[i] = u"“"
            elif char == "ca".decode("hex"):
                s[i] = u"”"
            elif char == "cb".decode("hex"):
                s[i] = u"‘"
            elif char == "a1".decode("hex"):
                s[i] = u"á"
            elif char == "a2".decode("hex"):
                s[i] = u"â"
            elif char == "a4".decode("hex"):
                s[i] = u"ä"
            elif char == "9f".decode("hex"):
                s[i] = u"ß"
            elif char == "a9".decode("hex"):
                s[i] = u"é"
            elif char == "a8".decode("hex"):
                s[i] = u"è"
            elif char == "aa".decode("hex"):
                s[i] = u"ê"
            elif char == "ac".decode("hex"):
                s[i] = u"ì"
            elif char == "ae".decode("hex"):
                s[i] = u"î"
            elif char == "b3".decode("hex"):
                s[i] = u"ó"
            elif char == "b4".decode("hex"):
                s[i] = u"ô"
            elif char == "b6".decode("hex"):
                s[i] = u"ö"
            elif char == "ba".decode("hex"):
                s[i] = u"ú"
            elif char == "bb".decode("hex"):
                s[i] = u"û"
            elif char == "bc".decode("hex"):
                s[i] = u"ü"
            elif char == "81".decode("hex"):
                s[i] = u"Á"
            elif char == "80".decode("hex"):
                s[i] = u"À"
            elif char == "82".decode("hex"):
                s[i] = u"Â"
            elif char == "84".decode("hex"):
                s[i] = u"Ä"
            elif char == "83".decode("hex"):
                s[i] = u"Ã"
            elif char == "87".decode("hex"):
                s[i] = u"Ç"
            elif char == "89".decode("hex"):
                s[i] = u"É"
            elif char == "88".decode("hex"):
                s[i] = u"È"
            elif char == "8a".decode("hex"):
                s[i] = u"Ê"
            elif char == "8b".decode("hex"):
                s[i] = u"Ë"
            elif char == "92".decode("hex"):
                s[i] = u"Œ"
            elif char == "8c".decode("hex"):
                s[i] = u"Ì"
            elif char == "8e".decode("hex"):
                s[i] = u"Î"
            elif char == "94".decode("hex"):
                s[i] = u"Ô"
            elif char == "96".decode("hex"):
                s[i] = u"Ö"
            elif char == "99".decode("hex"):
                s[i] = u"Ù"
            elif char == "9c".decode("hex"):
                s[i] = u"Ü"
            elif char == "b8".decode("hex"):
                s[i] = u"Ÿ"
            elif char == "a6".decode("hex"):
                s[i] = u"Φ"
            elif char == "b7".decode("hex"):
                s[i] = u"·"
            elif char == "a7".decode("hex"):
                s[i] = u"§"
            elif char == "a3".decode("hex"):
                s[i] = u"£"
            elif char == "b0".decode("hex"):
                s[i] = u"°"
            else:
                s[i] = unicode(char)
        except IndexError:
            break
        i += 1
    s = "".join(s)
    return s

def convertFromGreek (text):
    s = list(text)

    i = 0
    while True:
        try:
            char = s[i]
            if char.encode("hex") == "ce" or char.encode("hex") == "cf":
                s = s[0:i:] + s[i+1: :]
                i -= 1
            if char.encode("hex") == "b3":
                s[i] = "!"
            elif char.encode("hex") == "b8":
                s[i] = "@"
            elif char.encode("hex") == "bb":
                s[i] = "#"
            elif char.encode("hex") == "bc":
                s[i] = "$"
            elif char.encode("hex") == "bd":
                s[i] = "%"
            elif char.encode("hex") == "be":
                s[i] = "^"
            elif char.encode("hex") == "80":
                s[i] = "&"
            elif char.encode("hex") == "83":
                s[i] = "*"
            elif char.encode("hex") == "86":
                s[i] = "("
            elif char.encode("hex") == "88":
                s[i] = ")"
        except IndexError:
            break
        i += 1
    s = "".join(s)
    return s

def convertToGreek (text):
    s = list(text)

    i = 0
    while True:
        try:
            char = s[i]
            if char == "!":
                s[i] = "γ"
            elif char == "@":
                s[i] = "θ"
            elif char == "#":
                s[i] = "λ"
            elif char == "$":
                s[i] = "μ"
            elif char == "%":
                s[i] = "ν"
            elif char == "^":
                s[i] = "ξ"
            elif char == "&":
                s[i] = "π"
            elif char == "*":
                s[i] = "σ"
            elif char == "(":
                s[i] = "φ"
            elif char == ")":
                s[i] = "ψ"
        except IndexError:
            break
        i += 1
    s = "".join(s)
    return s
    
class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        '''
        if self._ignore_case:
            alphabet = alphabet.lower()
        '''

        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """

        if isinstance(text, str):
            #print(text)
            #text = convertFromGreek(text)
            text = [
                self.dict[char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def oneHot(v, v_length, nc):
    batchSize = v_length.size(0)
    maxLength = v_length.max()
    v_onehot = torch.FloatTensor(batchSize, maxLength, nc).fill_(0)
    acc = 0
    for i in range(batchSize):
        length = v_length[i]
        label = v[acc:acc + length].view(-1, 1).long()
        v_onehot[i, :length].scatter_(1, label, 1.0)
        acc += length
    return v_onehot


def loadData(v, data):
    with torch.no_grad():
        v.resize_(data.size()).copy_(data)


def prettyPrint(v):
    print('Size {0}, Type: {1}'.format(str(v.size()), v.data.type()))
    print('| Max: %f | Min: %f | Mean: %f' % (v.max().data[0], v.min().data[0],
                                              v.mean().data[0]))


def assureRatio(img):
    """Ensure imgH <= imgW."""
    b, c, h, w = img.size()
    if h > w:
        main = nn.UpsamplingBilinear2d(size=(h, h), scale_factor=None)
        img = main(img)
    return img
