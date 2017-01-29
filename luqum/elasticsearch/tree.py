from __future__ import absolute_import
import abc
import re


class JsonSerializableMixin(object):
    u"""
    Mixin to force subclasses to implement the json method
    """

    @property
    @abc.abstractmethod
    def json(self):
        pass  # pragma: no cover


class AbstractEItem(JsonSerializableMixin):
    u"""
    Base item element to build the "item" json
    For instance : {"term": {"field": {"value": "query"}}}
    """

    boost = None
    _fuzzy = None

    _KEYS_TO_ADD = (u'boost', u'fuzziness', )
    ADDITIONAL_KEYS_TO_ADD = ()

    def __init__(self, no_analyze=None, method=u'term', default_field=u'text'):
        self._method = method
        self._default_field = default_field
        self._fields = []
        self._no_analyze = no_analyze if no_analyze else []
        self.zero_terms_query = u'none'

    @property
    def json(self):

        inner_json = {}
        if self.method == u'query_string':
            json = {self.method: inner_json}
        else:
            json = {self.method: {self.field: inner_json}}

        # add base conf
        keys = self._KEYS_TO_ADD + self.ADDITIONAL_KEYS_TO_ADD
        for key in keys:
            value = getattr(self, key)
            if value is not None:
                if key == u'q' and self.method == u'match':
                    inner_json[u'query'] = value
                    inner_json[u'type'] = u'phrase'
                    inner_json[u'zero_terms_query'] = self.zero_terms_query
                elif key == u'q' and self.method == u'query_string':
                    inner_json[u'query'] = value
                    inner_json[u'analyze_wildcard'] = True
                    inner_json[u'default_field'] = self.field
                    inner_json[u'allow_leading_wildcard'] = True
                elif key == u'q':
                    inner_json[u'value'] = value
                else:
                    inner_json[key] = value
        return json

    @property
    def field(self):
        if self._fields:
            return u'.'.join(self._fields)
        else:
            return self._default_field

    def add_field(self, field):
        self._fields.insert(0, field)

    @property
    def fuzziness(self):
        return self._fuzzy

    @fuzziness.setter
    def fuzziness(self, fuzzy):
        self._method = u'fuzzy'
        self._fuzzy = fuzzy

    def _value_has_wildcard_char(self):
        return any(char in getattr(self, u'q', u'') for char in [u'*', u'?'])

    def _is_analyzed(self):
        return self.field in self._no_analyze

    @property
    def method(self):
        if self._is_analyzed() and self._value_has_wildcard_char():
            return u'wildcard'
        elif not self._is_analyzed() and self._value_has_wildcard_char():
            return u'query_string'
        elif not self._is_analyzed() and self._method == u'term':
            return u'match'
        return self._method


class EWord(AbstractEItem):
    u"""
    Build a word
    >>> from unittest import TestCase
    >>> TestCase().assertDictEqual(
    ...     EWord(q='test').json,
    ...     {'match': {'text': {
    ...         'zero_terms_query': 'none',
    ...         'type': 'phrase',
    ...         'query': 'test'
    ...     }}},
    ... )
    """

    ADDITIONAL_KEYS_TO_ADD = (u'q', )

    def __init__(self, q, *args, **kwargs):
        super(EWord, self).__init__(*args, **kwargs)
        self.q = q

    @property
    def json(self):
        if self.q == u'*':
            return {u"exists": {u"field": self.field}}
        return super(EWord, self).json


class EPhrase(AbstractEItem):
    u"""
    Build a phrase
    >>> from unittest import TestCase
    >>> TestCase().assertDictEqual(
    ...     EPhrase(phrase='"another test"').json,
    ...     {'match_phrase': {'text': {'query': 'another test'}}},
    ... )
    """

    ADDITIONAL_KEYS_TO_ADD = (u'query',)
    _proximity = None

    def __init__(self, phrase, *args, **kwargs):
        super(EPhrase, self).__init__(method=u'match_phrase', *args, **kwargs)
        phrase = self._replace_CR_and_LF_by_a_whitespace(phrase)
        self.query = self._remove_double_quotes(phrase)

    def __repr__(self):
        return u"%s(%s=%s)" % (self.__class__.__name__, self.field, self.query)

    def _replace_CR_and_LF_by_a_whitespace(self, phrase):
        return re.sub(ur'\s+', u' ', phrase)

    def _remove_double_quotes(self, phrase):
        return re.search(ur'"(?P<value>.+)"', phrase).group(u"value")

    @property
    def slop(self):
        return self._proximity

    @slop.setter
    def slop(self, slop):
        self._proximity = slop
        self.ADDITIONAL_KEYS_TO_ADD += (u'slop', )


class ERange(AbstractEItem):
    u"""
    Build a range
    >>> from unittest import TestCase
    >>> TestCase().assertDictEqual(
    ...     ERange(lt=100, gte=10).json,
    ...     {'range': {'text': {'lt': 100, 'gte': 10}}},
    ... )
    """

    def __init__(self, lt=None, lte=None, gt=None, gte=None, *args, **kwargs):
        super(ERange, self).__init__(method=u'range', *args, **kwargs)
        if lt and lt != u'*':
            self.lt = lt
            self.ADDITIONAL_KEYS_TO_ADD += (u'lt', )
        elif lte and lte != u'*':
            self.lte = lte
            self.ADDITIONAL_KEYS_TO_ADD += (u'lte', )
        if gt and gt != u'*':
            self.gt = gt
            self.ADDITIONAL_KEYS_TO_ADD += (u'gt', )
        elif gte and gte != u'*':
            self.gte = gte
            self.ADDITIONAL_KEYS_TO_ADD += (u'gte', )


class AbstractEOperation(JsonSerializableMixin):
    pass


class EOperation(AbstractEOperation):
    u"""
    Abstract operation taking care of the json build
    """

    def __init__(self, items):
        self.items = items
        self._method = None

    def __repr__(self):
        items = u", ".join(i.__repr__() for i in self.items)
        return u"%s(%s)" % (self.__class__.__name__, items)

    @property
    def json(self):
        return {u'bool': {self.operation: [item.json for item in self.items]}}


class ENested(AbstractEOperation):
    u"""
    Build ENested element

    Take care to remove ENested children
    """

    def __init__(self, nested_path, nested_fields, items, *args, **kwargs):

        self._nested_path = [nested_path]
        self.items = self._exclude_nested_children(items)

    @property
    def nested_path(self):
        return u'.'.join(self._nested_path)

    def add_nested_path(self, nested_path):
        self._nested_path.insert(0, nested_path)

    def __repr__(self):
        return u"%s(%s, %s)" % (self.__class__.__name__, self.nested_path, self.items)

    def _exclude_nested_children(self, subtree):
        u"""
        Rebuild tree excluding ENested in children if some are present

        >>> from unittest import TestCase
        >>> tree = EMust(items=[
        ...     ENested(
        ...         nested_path='a',
        ...         nested_fields=['a'],
        ...         items=EPhrase('"Francois"')
        ...     ),
        ...     ENested(
        ...         nested_path='a',
        ...         nested_fields=['a'],
        ...         items=EPhrase('"Dupont"'))
        ... ])
        >>> nested_node = ENested(
        ...     nested_path='a', nested_fields=['a'], items=tree)
        >>> TestCase().assertEqual(
        ...     nested_node.__repr__(),
        ...     'ENested(a, EMust(EPhrase(text=Francois), EPhrase(text=Dupont)))'
        ... )
        """
        if isinstance(subtree, ENested):
            # Exclude ENested

            if subtree.nested_path == self.nested_path:
                return self._exclude_nested_children(subtree.items)
            else:
                return subtree
        elif isinstance(subtree, AbstractEOperation):
            # Exclude ENested in children
            subtree.items = [
                self._exclude_nested_children(child)
                for child in subtree.items
            ]
            return subtree
        else:
            # return the subtree once ENested has been excluded
            return subtree

    @property
    def json(self):
        return {u'nested': {u'path': self.nested_path, u'query': self.items.json}}


class EShould(EOperation):
    u"""
    Build a should operation
    >>> from unittest import TestCase
    >>> json = EShould(
    ...     items=[EPhrase('"monty python"'), EPhrase('"spam eggs"')]
    ... ).json
    >>> TestCase().assertDictEqual(
    ...     json,
    ...     {'bool': {'should': [
    ...         {'match_phrase': {'text': {'query': 'monty python'}}},
    ...         {'match_phrase': {'text': {'query': 'spam eggs'}}},
    ...     ]}}
    ... )
    """
    operation = u'should'


class AbstractEMustOperation(EOperation):

    def __init__(self, items):
        op = super(AbstractEMustOperation, self).__init__(items)
        for item in self.items:
            item.zero_terms_query = self.zero_terms_query
        return op


class EMust(AbstractEMustOperation):
    u"""
    Build a must operation
    >>> from unittest import TestCase
    >>> json = EMust(
    ...     items=[EPhrase('"monty python"'), EPhrase('"spam eggs"')]
    ... ).json
    >>> TestCase().assertDictEqual(
    ...     json,
    ...     {'bool': {'must': [
    ...         {'match_phrase': {'text': {'query': 'monty python'}}},
    ...         {'match_phrase': {'text': {'query': 'spam eggs'}}},
    ...     ]}}
    ... )
    """
    zero_terms_query = u'all'
    operation = u'must'


class EMustNot(AbstractEMustOperation):
    u"""
    Build a must not operation
    >>> from unittest import TestCase
    >>> TestCase().assertDictEqual(
    ...     EMustNot(items=[EPhrase('"monty python"')]).json,
    ...     {'bool': {'must_not': [
    ...         {'match_phrase': {'text': {'query': 'monty python'}}},
    ...     ]}}
    ... )
    """
    zero_terms_query = u'none'
    operation = u'must_not'


class ElasticSearchItemFactory(object):
    u"""
    Factory to preconfigure EItems and EOperation
    At the moment, it's only used to pass the _no_analyze field
    >>> from unittest import TestCase
    >>> factory = ElasticSearchItemFactory(
    ...     no_analyze=['text'], nested_fields=[])
    >>> word = factory.build(EWord, q='test')
    >>> TestCase().assertDictEqual(
    ...     word.json,
    ...     {'term': {'text': {'value': 'test'}}},
    ... )
    """

    def __init__(self, no_analyze, nested_fields):
        self._no_analyze = no_analyze
        self._nested_fields = nested_fields

    def build(self, cls, *args, **kwargs):
        if issubclass(cls, AbstractEItem):
            return cls(
                no_analyze=self._no_analyze,
                *args,
                **kwargs
            )
        elif cls is ENested:
            return cls(
                nested_fields=self._nested_fields,
                *args,
                **kwargs
            )
        else:
            return cls(*args, **kwargs)
