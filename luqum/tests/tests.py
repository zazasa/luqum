# -*- coding: utf-8 -*-
from __future__ import with_statement
from __future__ import absolute_import
from unittest import TestCase

from luqum.exceptions import NestedSearchFieldException

from ..check import LuceneCheck, CheckNestedFields
from ..parser import lexer, parser, ParseError
from ..pretty import Prettifier, prettify
from ..tree import *
from ..utils import (
    LuceneTreeVisitor,
    LuceneTreeTransformer,
    LuceneTreeVisitorV2,
)


class TestTree(TestCase):

    def test_term_wildcard_true(self):
        self.assertTrue(Term(u"ba*").has_wildcard())
        self.assertTrue(Term(u"b*r").has_wildcard())
        self.assertTrue(Term(u"*ar").has_wildcard())

    def test_term_wildcard_false(self):
        self.assertFalse(Term(u"bar").has_wildcard())

    def test_term_is_only_a_wildcard(self):
        self.assertTrue(Term(u'*').is_wildcard())
        self.assertFalse(Term(u'*o').is_wildcard())
        self.assertFalse(Term(u'b*').is_wildcard())
        self.assertFalse(Term(u'b*o').is_wildcard())

    def test_equality_approx(self):
        u"""
        Regression test for a bug on approx equalities.
        Testing other tokens might be a good idea...
        """
        a1 = Proximity(term=u'foo', degree=5)
        a2 = Proximity(term=u'bar', degree=5)
        a3 = Proximity(term=u'foo', degree=5)

        self.assertNotEqual(a1, a2)
        self.assertEqual(a1, a3)

        f1 = Fuzzy(term=u'foo', degree=5)
        f2 = Fuzzy(term=u'bar', degree=5)
        f3 = Fuzzy(term=u'foo', degree=5)

        self.assertNotEqual(f1, f2)
        self.assertEqual(f1, f3)


class TestLexer(TestCase):
    u"""Test lexer
    """
    def test_basic(self):

        lexer.input(
            u'subject:test desc:(house OR car)^3 AND "big garage"~2 dirt~0.3 OR foo:{a TO z*]')
        self.assertEqual(lexer.token().value, Word(u"subject"))
        self.assertEqual(lexer.token().type, u"COLUMN")
        self.assertEqual(lexer.token().value, Word(u"test"))
        self.assertEqual(lexer.token().value, Word(u"desc"))
        self.assertEqual(lexer.token().type, u"COLUMN")
        self.assertEqual(lexer.token().type, u"LPAREN")
        self.assertEqual(lexer.token().value, Word(u"house"))
        self.assertEqual(lexer.token().type, u"OR_OP")
        self.assertEqual(lexer.token().value, Word(u"car"))
        self.assertEqual(lexer.token().type, u"RPAREN")
        t = lexer.token()
        self.assertEqual(t.type, u"BOOST")
        self.assertEqual(t.value, u"3")
        self.assertEqual(lexer.token().type, u"AND_OP")
        self.assertEqual(lexer.token().value, Phrase(u'"big garage"'))
        t = lexer.token()
        self.assertEqual(t.type, u"APPROX")
        self.assertEqual(t.value, u"2")
        self.assertEqual(lexer.token().value, Word(u"dirt"))
        t = lexer.token()
        self.assertEqual(t.type, u"APPROX")
        self.assertEqual(t.value, u"0.3")
        self.assertEqual(lexer.token().type, u"OR_OP")
        self.assertEqual(lexer.token().value, Word(u"foo"))
        self.assertEqual(lexer.token().type, u"COLUMN")
        self.assertEqual(lexer.token().type, u"LBRACKET")
        self.assertEqual(lexer.token().value, Word(u"a"))
        self.assertEqual(lexer.token().type, u"TO")
        self.assertEqual(lexer.token().value, Word(u"z*"))
        self.assertEqual(lexer.token().type, u"RBRACKET")
        self.assertEqual(lexer.token(), None)

    def test_accept_flavours(self):
        lexer.input(u'somedate:[now/d-1d+7H TO now/d+7H]')

        self.assertEqual(lexer.token().value, Word(u'somedate'))

        self.assertEqual(lexer.token().type, u"COLUMN")
        self.assertEqual(lexer.token().type, u"LBRACKET")

        self.assertEqual(lexer.token().value, Word(u"now/d-1d+7H"))
        self.assertEqual(lexer.token().type, u"TO")
        self.assertEqual(lexer.token().value, Word(u"now/d+7H"))

        self.assertEqual(lexer.token().type, u"RBRACKET")


class TestParser(TestCase):
    u"""Test base parser

    .. note:: we compare str(tree) before comparing tree, because it's more easy to debug
    """

    def test_simplest(self):
        tree = (
            AndOperation(
                Word(u"foo"),
                Word(u"bar")))
        parsed = parser.parse(u"foo AND bar")
        self.assertEqual(unicode(parsed), unicode(tree))
        self.assertEqual(parsed, tree)

    def test_implicit_operations(self):
        tree = (
            UnknownOperation(
                Word(u"foo"),
                Word(u"bar")))
        parsed = parser.parse(u"foo bar")
        self.assertEqual(unicode(parsed), unicode(tree))
        self.assertEqual(parsed, tree)

    def test_simple_field(self):
        tree = (
            SearchField(
                u"subject",
                Word(u"test")))
        parsed = parser.parse(u"subject:test")
        self.assertEqual(unicode(parsed), unicode(tree))
        self.assertEqual(parsed, tree)

    def test_minus(self):
        tree = (
            AndOperation(
                AndOperation(
                    Prohibit(
                        Word(u"test")),
                    Prohibit(
                        Word(u"foo"))),
                Not(
                    Word(u"bar"))))
        parsed = parser.parse(u"-test AND -foo AND NOT bar")
        self.assertEqual(unicode(parsed), unicode(tree))
        self.assertEqual(parsed, tree)

    def test_plus(self):
        tree = (
            AndOperation(
                AndOperation(
                    Plus(
                        Word(u"test")),
                    Word(u"foo")),
                Plus(
                    Word(u"bar"))))
        parsed = parser.parse(u"+test AND foo AND +bar")
        self.assertEqual(unicode(parsed), unicode(tree))
        self.assertEqual(parsed, tree)

    def test_phrase(self):
        tree = (
            AndOperation(
                Phrase(u'"a phrase (AND a complicated~ one)"'),
                Phrase(u'"Another one"')))
        parsed = parser.parse(u'"a phrase (AND a complicated~ one)" AND "Another one"')
        self.assertEqual(unicode(parsed), unicode(tree))
        self.assertEqual(parsed, tree)

    def test_approx(self):
        tree = (
            UnknownOperation(
                Proximity(
                    Phrase(u'"foo bar"'),
                    3),
                UnknownOperation(
                    Proximity(
                        Phrase(u'"foo baz"'),
                        1),
                    UnknownOperation(
                        Fuzzy(
                            Word(u'baz'),
                            Decimal(u"0.3")),
                        Fuzzy(
                            Word(u'fou'),
                            Decimal(u"0.5"))))))
        parsed = parser.parse(u'"foo bar"~3 "foo baz"~ baz~0.3 fou~')
        self.assertEqual(unicode(parsed), unicode(tree))
        self.assertEqual(parsed, tree)

    def test_boost(self):
        tree = (
            UnknownOperation(
                Boost(
                    Phrase(u'"foo bar"'),
                    Decimal(u"3.0")),
                Boost(
                    Group(
                        AndOperation(
                            Word(u'baz'),
                            Word(u'bar'))),
                    Decimal(u"2.1"))))
        parsed = parser.parse(u'"foo bar"^3 (baz AND bar)^2.1')
        self.assertEqual(unicode(parsed), unicode(tree))
        self.assertEqual(parsed, tree)

    def test_groups(self):
        tree = (
           OrOperation(
               Word(u'test'),
               Group(
                   AndOperation(
                       SearchField(
                           u"subject",
                           FieldGroup(
                               OrOperation(
                                   Word(u'foo'),
                                   Word(u'bar')))),
                       Word(u'baz')))))
        parsed = parser.parse(u'test OR (subject:(foo OR bar) AND baz)')
        self.assertEqual(unicode(parsed), unicode(tree))
        self.assertEqual(parsed, tree)

    def test_range(self):
        tree = (
            AndOperation(
                SearchField(
                    u"foo",
                    Range(Word(u"10"), Word(u"100"), True, True)),
                SearchField(
                    u"bar",
                    Range(Word(u"a*"), Word(u"*"), True, False))))
        parsed = parser.parse(u'foo:[10 TO 100] AND bar:[a* TO *}')
        self.assertEqual(unicode(parsed), unicode(tree))
        self.assertEqual(parsed, tree)

    def test_flavours(self):
        tree = SearchField(
            u"somedate",
            Range(Word(u"now/d-1d+7H"), Word(u"now/d+7H"), True, True))
        parsed = parser.parse(u'somedate:[now/d-1d+7H TO now/d+7H]')
        self.assertEqual(unicode(parsed), unicode(tree))
        self.assertEqual(parsed, tree)

    def test_combinations(self):
        # self.assertEqual(parser.parse("subject:test desc:(house OR car)").pval, "")
        tree = (
            UnknownOperation(
                SearchField(
                    u"subject",
                    Word(u"test")),
                AndOperation(
                    SearchField(
                        u"desc",
                        FieldGroup(
                            OrOperation(
                                Word(u"house"),
                                Word(u"car")))),
                    Not(
                        Proximity(
                            Phrase(u'"approximatly this"'),
                            3)))))
        parsed = parser.parse(u'subject:test desc:(house OR car) AND NOT "approximatly this"~3')

        self.assertEqual(unicode(parsed), unicode(tree))
        self.assertEqual(parsed, tree)

    def test_reserved_ok(self):
        u"""Test reserved word do not hurt in certain positions
        """
        tree = SearchField(u"foo", Word(u"TO"))
        parsed = parser.parse(u'foo:TO')
        self.assertEqual(unicode(tree), unicode(parsed))
        self.assertEqual(tree, parsed)
        tree = SearchField(u"foo", Word(u"TO*"))
        parsed = parser.parse(u'foo:TO*')
        self.assertEqual(unicode(tree), unicode(parsed))
        self.assertEqual(tree, parsed)
        tree = SearchField(u"foo", Word(u"NOT*"))
        parsed = parser.parse(u'foo:NOT*')
        self.assertEqual(unicode(tree), unicode(parsed))
        self.assertEqual(tree, parsed)
        tree = SearchField(u"foo", Phrase(u'"TO AND OR"'))
        parsed = parser.parse(u'foo:"TO AND OR"')
        self.assertEqual(unicode(tree), unicode(parsed))
        self.assertEqual(tree, parsed)

    def test_date_in_field(self):
        tree = SearchField(u"foo", Word(u"2015-12-19"))
        parsed = parser.parse(u'foo:2015-12-19')
        self.assertEqual(unicode(tree), unicode(parsed))
        self.assertEqual(tree, parsed)
        tree = SearchField(u"foo", Word(u"2015-12-19T22:30"))
        parsed = parser.parse(u'foo:2015-12-19T22:30')
        self.assertEqual(unicode(tree), unicode(parsed))
        self.assertEqual(tree, parsed)
        tree = SearchField(u"foo", Word(u"2015-12-19T22:30:45"))
        parsed = parser.parse(u'foo:2015-12-19T22:30:45')
        self.assertEqual(unicode(tree), unicode(parsed))
        self.assertEqual(tree, parsed)
        tree = SearchField(u"foo", Word(u"2015-12-19T22:30:45.234Z"))
        parsed = parser.parse(u'foo:2015-12-19T22:30:45.234Z')
        self.assertEqual(unicode(tree), unicode(parsed))
        self.assertEqual(tree, parsed)

    def test_datemath_in_field(self):
        tree = SearchField(u"foo", Word(ur"2015-12-19||+2\d"))
        parsed = parser.parse(ur'foo:2015-12-19||+2\d')
        self.assertEqual(unicode(tree), unicode(parsed))
        self.assertEqual(tree, parsed)
        tree = SearchField(u"foo", Word(ur"now+2h+20m\h"))
        parsed = parser.parse(ur'foo:now+2h+20m\h')
        self.assertEqual(unicode(tree), unicode(parsed))
        self.assertEqual(tree, parsed)

    def test_date_in_range(self):
        # juste one funky expression
        tree = SearchField(u"foo", Range(Word(ur"2015-12-19||+2\d"), Word(ur"now+3d+12h\h")))
        parsed = parser.parse(ur'foo:[2015-12-19||+2\d TO now+3d+12h\h]')
        self.assertEqual(unicode(tree), unicode(parsed))
        self.assertEqual(tree, parsed)

    def test_reserved_ko(self):
        u"""Test reserved word hurt as they hurt lucene
        """
        with self.assertRaises(ParseError):
            parser.parse(u'foo:NOT')
        with self.assertRaises(ParseError):
            parser.parse(u'foo:AND')
        with self.assertRaises(ParseError):
            parser.parse(u'foo:OR')
        with self.assertRaises(ParseError):
            parser.parse(u'OR')
        with self.assertRaises(ParseError):
            parser.parse(u'AND')


class TestPrettify(TestCase):

    big_tree = AndOperation(
        Group(OrOperation(Word(u"baaaaaaaaaar"), Word(u"baaaaaaaaaaaaaz"))), Word(u"fooooooooooo"))
    fat_tree = AndOperation(
        SearchField(
            u"subject",
            FieldGroup(
                OrOperation(
                    Word(u"fiiiiiiiiiiz"),
                    AndOperation(Word(u"baaaaaaaaaar"), Word(u"baaaaaaaaaaaaaz"))))),
        AndOperation(Word(u"fooooooooooo"), Word(u"wiiiiiiiiiz")))

    def test_one_liner(self):
        tree = AndOperation(Group(OrOperation(Word(u"bar"), Word(u"baz"))), Word(u"foo"))
        self.assertEqual(prettify(tree), u"( bar OR baz ) AND foo")

    def test_small(self):
        prettify = Prettifier(indent=8, max_len=20)
        self.assertEqual(
            u"\n" + prettify(self.big_tree), u"""
(
        baaaaaaaaaar
        OR
        baaaaaaaaaaaaaz
)
AND
fooooooooooo""")
        self.assertEqual(
            u"\n" + prettify(self.fat_tree), u"""
subject: (
        fiiiiiiiiiiz
        OR
                baaaaaaaaaar
                AND
                baaaaaaaaaaaaaz
)
AND
fooooooooooo
AND
wiiiiiiiiiz""")

    def test_small_inline_ops(self):
        prettify = Prettifier(indent=8, max_len=20, inline_ops=True)
        self.assertEqual(u"\n" + prettify(self.big_tree), u"""
(
        baaaaaaaaaar OR
        baaaaaaaaaaaaaz ) AND
fooooooooooo""")
        self.assertEqual(u"\n" + prettify(self.fat_tree), u"""
subject: (
        fiiiiiiiiiiz OR
                baaaaaaaaaar AND
                baaaaaaaaaaaaaz ) AND
fooooooooooo AND
wiiiiiiiiiz""")

    def test_normal(self):
        prettify = Prettifier(indent=4, max_len=50)
        self.assertEqual(u"\n" + prettify(self.big_tree), u"""
(
    baaaaaaaaaar OR baaaaaaaaaaaaaz
)
AND
fooooooooooo""")
        self.assertEqual(u"\n" + prettify(self.fat_tree), u"""
subject: (
    fiiiiiiiiiiz
    OR
        baaaaaaaaaar AND baaaaaaaaaaaaaz
)
AND
fooooooooooo
AND
wiiiiiiiiiz""")

    def test_normal_inline_ops(self):
        prettify = Prettifier(indent=4, max_len=50, inline_ops=True)
        self.assertEqual(u"\n" + prettify(self.big_tree), u"""
(
    baaaaaaaaaar OR baaaaaaaaaaaaaz ) AND
fooooooooooo""")
        self.assertEqual(u"\n" + prettify(self.fat_tree), u"""
subject: (
    fiiiiiiiiiiz OR
        baaaaaaaaaar AND baaaaaaaaaaaaaz ) AND
fooooooooooo AND
wiiiiiiiiiz""")


class TestCheck(TestCase):

    def test_check_ok(self):
        query = (
            AndOperation(
                SearchField(
                    u"f",
                    FieldGroup(
                        AndOperation(
                            Boost(Proximity(Phrase(u'"foo bar"'), 4), u"4.2"),
                            Prohibit(Range(u"100", u"200"))))),
                Group(
                    OrOperation(
                        Fuzzy(Word(u"baz"), u".8"),
                        Plus(Word(u"fizz"))))))
        check = LuceneCheck()
        self.assertTrue(check(query))
        self.assertEqual(check.errors(query), [])

    def test_bad_fieldgroup(self):
        check = LuceneCheck()
        query = FieldGroup(Word(u"foo"))
        self.assertFalse(check(query))
        self.assertEqual(len(check.errors(query)), 1)
        self.assertIn(u"FieldGroup misuse", check.errors(query)[0])

        query = OrOperation(
            FieldGroup(Word(u"bar")),
            Word(u"foo"))
        self.assertFalse(check(query))
        self.assertEqual(len(check.errors(query)), 1)
        self.assertIn(u"FieldGroup misuse", check.errors(query)[0])

    def test_bad_group(self):
        check = LuceneCheck()
        query = SearchField(u"f", Group(Word(u"foo")))
        self.assertFalse(check(query))
        self.assertEqual(len(check.errors(query)), 2)  # one for bad expr, one for misuse
        self.assertIn(u"Group misuse", u"".join(check.errors(query)))

    def test_zealous_or_not(self):
        query = (
            OrOperation(
                Prohibit(Word(u"foo")),
                Word(u"bar")))
        check_zealous = LuceneCheck(zeal=1)
        self.assertFalse(check_zealous(query))
        self.assertIn(u"inconsistent", check_zealous.errors(query)[0])
        check_easy_going = LuceneCheck()
        self.assertTrue(check_easy_going(query))

    def test_bad_field_name(self):
        check = LuceneCheck()
        query = SearchField(u"foo*", Word(u"bar"))
        self.assertFalse(check(query))
        self.assertEqual(len(check.errors(query)), 1)
        self.assertIn(u"not a valid field name", check.errors(query)[0])

    def test_bad_field_expr(self):
        check = LuceneCheck()
        query = SearchField(u"foo", Prohibit(Word(u"bar")))
        self.assertFalse(check(query))
        self.assertEqual(len(check.errors(query)), 1)
        self.assertIn(u"not valid", check.errors(query)[0])

    def test_word_space(self):
        check = LuceneCheck()
        query = Word(u"foo bar")
        self.assertFalse(check(query))
        self.assertEqual(len(check.errors(query)), 1)
        self.assertIn(u"space", check.errors(query)[0])

    def test_invalid_characters_in_word_value(self):
        query = Word(u"foo/bar")
        # Passes if zeal == 0
        check = LuceneCheck()
        self.assertTrue(check(query))
        self.assertEqual(len(check.errors(query)), 0)
        # But not if zeal == 1
        check = LuceneCheck(zeal=1)
        self.assertFalse(check(query))
        self.assertEqual(len(check.errors(query)), 1)
        self.assertIn(u"Invalid characters", check.errors(query)[0])

    def test_fuzzy_negative_degree(self):
        check = LuceneCheck()
        query = Fuzzy(Word(u"foo"), u"-4.1")
        self.assertFalse(check(query))
        self.assertEqual(len(check.errors(query)), 1)
        self.assertIn(u"invalid degree", check.errors(query)[0])

    def test_fuzzy_non_word(self):
        check = LuceneCheck()
        query = Fuzzy(Phrase(u'"foo bar"'), u"2")
        self.assertFalse(check(query))
        self.assertEqual(len(check.errors(query)), 1)
        self.assertIn(u"single term", check.errors(query)[0])

    def test_proximity_non_phrase(self):
        check = LuceneCheck()
        query = Proximity(Word(u"foo"), u"2")
        self.assertFalse(check(query))
        self.assertEqual(len(check.errors(query)), 1)
        self.assertIn(u"phrase", check.errors(query)[0])

    def test_unknown_item_type(self):
        check = LuceneCheck()
        query = AndOperation(u"foo", 2)
        self.assertFalse(check(query))
        self.assertEqual(len(check.errors(query)), 2)
        self.assertIn(u"Unknown item type", check.errors(query)[0])
        self.assertIn(u"Unknown item type", check.errors(query)[1])


class TreeVisitorTestCase(TestCase):

    class BasicVisitor(LuceneTreeVisitor):
        u""" Dummy visitor, simply yielding a list of nodes. """
        def generic_visit(self, node, parents):
            yield node

    class MROVisitor(LuceneTreeVisitor):

        def visit_or_operation(self, node, parents=[]):
            return [u"{} OR {}".format(*node.children)]

        def visit_base_operation(self, node, parents=[]):
            return [u"{} BASE_OP {}".format(*node.children)]

        def visit_word(self, node, parents=[]):
            return [node.value]

    def test_generic_visit(self):
        tree = (
            AndOperation(
                Word(u"foo"),
                Word(u"bar")))

        visitor = LuceneTreeVisitor()
        nodes = list(visitor.visit(tree))
        self.assertEqual(nodes, [])

    def test_basic_traversal(self):
        tree = (
            AndOperation(
                Word(u"foo"),
                Word(u"bar")))

        visitor = self.BasicVisitor()
        nodes = list(visitor.visit(tree))

        self.assertListEqual(
            [AndOperation(Word(u'foo'), Word(u'bar')), Word(u'foo'), Word(u'bar')],
            nodes)

    def test_mro(self):
        visitor = self.MROVisitor()

        tree = OrOperation(Word(u'a'), Word(u'b'))
        result = visitor.visit(tree)
        self.assertEquals(list(result), [u'a OR b', u'a', u'b'])

        tree = AndOperation(Word(u'a'), Word(u'b'))
        result = visitor.visit(tree)
        self.assertEquals(list(result), [u'a BASE_OP b', u'a', u'b'])


class TreeTransformerTestCase(TestCase):

    class BasicTransformer(LuceneTreeTransformer):
        u"""
        Dummy transformer that simply turn any Word node's value into "lol"
        """
        def visit_word(self, node, parent):
            return Word(u'lol')

    def test_basic_traversal(self):
        tree = (
            AndOperation(
                Word(u"foo"),
                Word(u"bar")))

        transformer = self.BasicTransformer()
        new_tree = transformer.visit(tree)

        self.assertEqual(
            (AndOperation(
                Word(u"lol"),
                Word(u"lol"))), new_tree)


class TreeVisitorV2TestCase(TestCase):

    class BasicVisitor(LuceneTreeVisitorV2):
        u""" Dummy visitor, simply yielding a list of nodes. """
        def generic_visit(self, node, parents, context):
            yield node
            for c in node.children:
                for item in self.visit(c, parents + [node], context):
                    yield item

    class MROVisitor(LuceneTreeVisitorV2):

        def visit_or_operation(self, node, parents=[], context=None):
            return u"{} OR {}".format(*[self.visit(c) for c in node.children])

        def visit_base_operation(self, node, parents=[], context=None):
            return u"{} BASE_OP {}".format(*[self.visit(c) for c in node.children])

        def visit_word(self, node, parents=[], context=None):
            return node.value

    def test_basic_traversal(self):
        tree = (
            AndOperation(
                Word(u"foo"),
                Word(u"bar")))

        visitor = self.BasicVisitor()
        nodes = list(visitor.visit(tree))

        self.assertListEqual(
            [AndOperation(Word(u'foo'), Word(u'bar')), Word(u'foo'), Word(u'bar')],
            nodes)

    def test_mro(self):
        visitor = self.MROVisitor()

        tree = OrOperation(Word(u'a'), Word(u'b'))
        result = visitor.visit(tree)
        self.assertEquals(result, u'a OR b')

        tree = OrOperation(AndOperation(Word(u'a'), Word(u'b')), Word(u'c'))
        result = visitor.visit(tree)
        self.assertEquals(result, u'a BASE_OP b OR c')

    def test_generic_visit_fails_by_default(self):
        visitor = self.MROVisitor()
        with self.assertRaises(AttributeError):
            visitor.visit(Phrase(u'"test"'))


class CheckVisitorTestCase(TestCase):

    def setUp(self):

        NESTED_FIELDS = {
            u'author': {
                u'firstname': {},
                u'book': {
                    u'title': {},
                    u'format': {
                        u'type': {}
                    }
                }
            },
        }

        self.checker = CheckNestedFields(nested_fields=NESTED_FIELDS)

    def test_correct_nested_lucene_query_wo_point_not_raise(self):
        tree = parser.parse(u'author:book:title:"foo" AND '
                            u'author:book:format:type: "pdf"')
        self.checker(tree)
        self.assertIsNotNone(tree)

    def test_correct_nested_lucene_query_with_point_not_raise(self):
        tree = parser.parse(u'author.book.title:"foo" AND '
                            u'author.book.format.type:"pdf"')
        self.checker(tree)
        self.assertIsNotNone(tree)

    def test_incorrect_nested_lucene_query_wo_point_raise(self):
        tree = parser.parse(u'author:gender:"Mr" AND '
                            u'author:book:format:type:"pdf"')
        with self.assertRaises(NestedSearchFieldException) as e:
            self.checker(tree)
        self.assertIn(u'"gender"', unicode(e.exception))

    def test_incorrect_nested_lucene_query_with_point_raise(self):
        tree = parser.parse(u'author.gender:"Mr" AND '
                            u'author.book.format.type:"pdf"')
        with self.assertRaises(NestedSearchFieldException) as e:
            self.checker(tree)
        self.assertIn(u'"gender"', unicode(e.exception))

    def test_correct_nested_lucene_query_with_and_wo_point_not_raise(self):
        tree = parser.parse(
            u'author:(book.title:"foo" OR book.title:"bar")')
        self.checker(tree)
        self.assertIsNotNone(tree)

    def test_simple_query_with_a_nested_field_should_raise(self):
        tree = parser.parse(u'author:"foo"')
        with self.assertRaises(NestedSearchFieldException) as e:
            self.checker(tree)
        self.assertIn(u'"author"', unicode(e.exception))

    def test_simple_query_with_a_multi_nested_field_should_raise(self):
        tree = parser.parse(u'author:book:"foo"')
        with self.assertRaises(NestedSearchFieldException) as e:
            self.checker(tree)
        self.assertIn(u'"book"', unicode(e.exception))

    def test_complex_query_with_a_multi_nested_field_should_raise(self):
        tree = parser.parse(u'author:test OR author.firstname:"Hugo"')
        with self.assertRaises(NestedSearchFieldException) as e:
            self.checker(tree)
        self.assertIn(u'"author"', unicode(e.exception))

    def test_complex_query_wo_point_with_a_multi_nested_field_should_raise(self):
        tree = parser.parse(u'author:("test" AND firstname:Hugo)')
        with self.assertRaises(NestedSearchFieldException) as e:
            self.checker(tree)
        self.assertIn(u'"author"', unicode(e.exception))
