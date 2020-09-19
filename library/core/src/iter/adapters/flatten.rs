use crate::fmt;
use crate::ops::Try;

use super::super::{DoubleEndedIterator, Fuse, FusedIterator, Iterator};
use super::Map;

/// An iterator that maps each element to an iterator, and yields the elements
/// of the produced iterators.
///
/// This `struct` is created by [`Iterator::flat_map`]. See its documentation
/// for more.
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct FlatMap<I, U: IntoIterator, F> {
    inner: FlattenCompat<Map<I, F>, <U as IntoIterator>::IntoIter>,
}
impl<I: Iterator, U: IntoIterator, F: FnMut(I::Item) -> U> FlatMap<I, U, F> {
    pub(in super::super) fn new(iter: I, f: F) -> FlatMap<I, U, F> {
        FlatMap { inner: FlattenCompat::new(iter.map(f)) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: Clone, U, F: Clone> Clone for FlatMap<I, U, F>
where
    U: Clone + IntoIterator<IntoIter: Clone>,
{
    fn clone(&self) -> Self {
        FlatMap { inner: self.inner.clone() }
    }
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<I: fmt::Debug, U, F> fmt::Debug for FlatMap<I, U, F>
where
    U: IntoIterator<IntoIter: fmt::Debug>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FlatMap").field("inner", &self.inner).finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: Iterator, U: IntoIterator, F> Iterator for FlatMap<I, U, F>
where
    F: FnMut(I::Item) -> U,
{
    type Item = U::Item;

    #[inline]
    fn next(&mut self) -> Option<U::Item> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    #[inline]
    fn try_fold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        self.inner.try_fold(init, fold)
    }

    #[inline]
    fn fold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.inner.fold(init, fold)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: DoubleEndedIterator, U, F> DoubleEndedIterator for FlatMap<I, U, F>
where
    F: FnMut(I::Item) -> U,
    U: IntoIterator<IntoIter: DoubleEndedIterator>,
{
    #[inline]
    fn next_back(&mut self) -> Option<U::Item> {
        self.inner.next_back()
    }

    #[inline]
    fn try_rfold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        self.inner.try_rfold(init, fold)
    }

    #[inline]
    fn rfold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.inner.rfold(init, fold)
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<I, U, F> FusedIterator for FlatMap<I, U, F>
where
    I: FusedIterator,
    U: IntoIterator,
    F: FnMut(I::Item) -> U,
{
}

/// An iterator that flattens one level of nesting in an iterator of things
/// that can be turned into iterators.
///
/// This `struct` is created by the [`flatten`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`flatten`]: trait.Iterator.html#method.flatten
/// [`Iterator`]: trait.Iterator.html
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "iterator_flatten", since = "1.29.0")]
pub struct Flatten<I: Iterator<Item: IntoIterator>> {
    inner: FlattenCompat<I, <I::Item as IntoIterator>::IntoIter>,
}

impl<I: Iterator<Item: IntoIterator>> Flatten<I> {
    pub(in super::super) fn new(iter: I) -> Flatten<I> {
        Flatten { inner: FlattenCompat::new(iter) }
    }
}

#[stable(feature = "iterator_flatten", since = "1.29.0")]
impl<I, U> fmt::Debug for Flatten<I>
where
    I: fmt::Debug + Iterator<Item: IntoIterator<IntoIter = U, Item = U::Item>>,
    U: fmt::Debug + Iterator,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Flatten").field("inner", &self.inner).finish()
    }
}

#[stable(feature = "iterator_flatten", since = "1.29.0")]
impl<I, U> Clone for Flatten<I>
where
    I: Clone + Iterator<Item: IntoIterator<IntoIter = U, Item = U::Item>>,
    U: Clone + Iterator,
{
    fn clone(&self) -> Self {
        Flatten { inner: self.inner.clone() }
    }
}

#[stable(feature = "iterator_flatten", since = "1.29.0")]
impl<I, U> Iterator for Flatten<I>
where
    I: Iterator<Item: IntoIterator<IntoIter = U, Item = U::Item>>,
    U: Iterator,
{
    type Item = U::Item;

    #[inline]
    fn next(&mut self) -> Option<U::Item> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    #[inline]
    fn try_fold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        self.inner.try_fold(init, fold)
    }

    #[inline]
    fn fold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.inner.fold(init, fold)
    }
}

#[stable(feature = "iterator_flatten", since = "1.29.0")]
impl<I, U> DoubleEndedIterator for Flatten<I>
where
    I: DoubleEndedIterator<Item: IntoIterator<IntoIter = U, Item = U::Item>>,
    U: DoubleEndedIterator,
{
    #[inline]
    fn next_back(&mut self) -> Option<U::Item> {
        self.inner.next_back()
    }

    #[inline]
    fn try_rfold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        self.inner.try_rfold(init, fold)
    }

    #[inline]
    fn rfold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.inner.rfold(init, fold)
    }
}

#[stable(feature = "iterator_flatten", since = "1.29.0")]
impl<I, U> FusedIterator for Flatten<I>
where
    I: FusedIterator<Item: IntoIterator<IntoIter = U, Item = U::Item>>,
    U: Iterator,
{
}

/// Real logic of both `Flatten` and `FlatMap` which simply delegate to
/// this type.
#[derive(Clone, Debug)]
struct FlattenCompat<I, U> {
    iter: Fuse<I>,
    frontiter: Option<U>,
    backiter: Option<U>,
}
impl<I, U> FlattenCompat<I, U>
where
    I: Iterator,
{
    /// Adapts an iterator by flattening it, for use in `flatten()` and `flat_map()`.
    fn new(iter: I) -> FlattenCompat<I, U> {
        FlattenCompat { iter: iter.fuse(), frontiter: None, backiter: None }
    }
}

impl<I, U> FlattenCompat<I, U>
where
    I: Iterator<Item: IntoIterator<IntoIter = U, Item = U::Item>>,
    U: Iterator,
{
    /// Folds over the inner iterators.
    #[inline]
    fn iter_try_fold<Acc, Fold, R>(&mut self, mut init: Acc, mut fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, &mut U) -> R,
        R: Try<Ok = Acc>,
    {
        fn flatten<'a, T: IntoIterator, Acc, R: Try<Ok = Acc>>(
            frontiter: &'a mut Option<T::IntoIter>,
            fold: &'a mut impl FnMut(Acc, &mut T::IntoIter) -> R,
        ) -> impl FnMut(Acc, T) -> R + 'a {
            move |acc, iter| {
                let mut mid = iter.into_iter();
                let r = fold(acc, &mut mid);
                *frontiter = Some(mid);
                r
            }
        }

        if let Some(ref mut iter) = self.frontiter {
            init = fold(init, iter)?;
        }
        self.frontiter = None;

        init = self.iter.try_fold(init, flatten(&mut self.frontiter, &mut fold))?;
        self.frontiter = None;

        if let Some(ref mut iter) = self.backiter {
            init = fold(init, iter)?;
        }
        self.backiter = None;

        Try::from_ok(init)
    }
}

impl<I, U> FlattenCompat<I, U>
where
    I: DoubleEndedIterator<Item: IntoIterator<IntoIter = U, Item = U::Item>>,
    U: DoubleEndedIterator,
{
    /// Folds over the inner iterators in reverse order.
    #[inline]
    fn iter_try_rfold<Acc, Fold, R>(&mut self, mut init: Acc, mut fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, &mut U) -> R,
        R: Try<Ok = Acc>,
    {
        fn flatten<'a, T: IntoIterator, Acc, R: Try<Ok = Acc>>(
            backiter: &'a mut Option<T::IntoIter>,
            fold: &'a mut impl FnMut(Acc, &mut T::IntoIter) -> R,
        ) -> impl FnMut(Acc, T) -> R + 'a {
            move |acc, iter| {
                let mut mid = iter.into_iter();
                let r = fold(acc, &mut mid);
                *backiter = Some(mid);
                r
            }
        }

        if let Some(ref mut iter) = self.backiter {
            init = fold(init, iter)?;
        }
        self.backiter = None;

        init = self.iter.try_rfold(init, flatten(&mut self.backiter, &mut fold))?;
        self.backiter = None;

        if let Some(ref mut iter) = self.frontiter {
            init = fold(init, iter)?;
        }
        self.frontiter = None;

        Try::from_ok(init)
    }
}

impl<I, U> Iterator for FlattenCompat<I, U>
where
    I: Iterator<Item: IntoIterator<IntoIter = U, Item = U::Item>>,
    U: Iterator,
{
    type Item = U::Item;

    #[inline]
    fn next(&mut self) -> Option<U::Item> {
        loop {
            if let Some(ref mut inner) = self.frontiter {
                match inner.next() {
                    None => self.frontiter = None,
                    elt @ Some(_) => return elt,
                }
            }
            match self.iter.next() {
                None => return self.backiter.as_mut()?.next(),
                Some(inner) => self.frontiter = Some(inner.into_iter()),
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (flo, fhi) = self.frontiter.as_ref().map_or((0, Some(0)), U::size_hint);
        let (blo, bhi) = self.backiter.as_ref().map_or((0, Some(0)), U::size_hint);
        let lo = flo.saturating_add(blo);
        match (self.iter.size_hint(), fhi, bhi) {
            ((0, Some(0)), Some(a), Some(b)) => (lo, a.checked_add(b)),
            _ => (lo, None),
        }
    }

    #[inline]
    fn advance_by(&mut self, n: usize) -> Result<(), usize> {
        fn advance<I: Iterator>(n: usize, iter: &mut I) -> Option<usize> {
            match iter.advance_by(n) {
                Ok(()) => None,
                Err(k) => Some(n - k),
            }
        }

        match self.iter_try_fold(n, advance) {
            Some(rem) => Err(n - rem),
            None => Ok(()),
        }
    }

    #[inline]
    fn try_fold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        #[inline]
        fn flatten<I: Iterator, Acc, R: Try<Ok = Acc>>(
            mut fold: impl FnMut(Acc, I::Item) -> R,
        ) -> impl FnMut(Acc, &mut I) -> R {
            move |acc, iter| iter.try_fold(acc, &mut fold)
        }

        self.iter_try_fold(init, flatten(fold))
    }

    #[inline]
    fn fold<Acc, Fold>(self, init: Acc, ref mut fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        #[inline]
        fn flatten<U: Iterator, Acc>(
            fold: &mut impl FnMut(Acc, U::Item) -> Acc,
        ) -> impl FnMut(Acc, U) -> Acc + '_ {
            move |acc, iter| iter.fold(acc, &mut *fold)
        }

        self.frontiter
            .into_iter()
            .chain(self.iter.map(IntoIterator::into_iter))
            .chain(self.backiter)
            .fold(init, flatten(fold))
    }
}

impl<I, U> DoubleEndedIterator for FlattenCompat<I, U>
where
    I: DoubleEndedIterator<Item: IntoIterator<IntoIter = U, Item = U::Item>>,
    U: DoubleEndedIterator,
{
    #[inline]
    fn next_back(&mut self) -> Option<U::Item> {
        loop {
            if let Some(ref mut inner) = self.backiter {
                match inner.next_back() {
                    None => self.backiter = None,
                    elt @ Some(_) => return elt,
                }
            }
            match self.iter.next_back() {
                None => return self.frontiter.as_mut()?.next_back(),
                next => self.backiter = next.map(IntoIterator::into_iter),
            }
        }
    }

    #[inline]
    fn advance_back_by(&mut self, n: usize) -> Result<(), usize> {
        fn advance<I: DoubleEndedIterator>(n: usize, iter: &mut I) -> Option<usize> {
            match iter.advance_back_by(n) {
                Ok(()) => None,
                Err(k) => Some(n - k),
            }
        }

        match self.iter_try_rfold(n, advance) {
            Some(rem) => Err(n - rem),
            None => Ok(()),
        }
    }

    #[inline]
    fn try_rfold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        #[inline]
        fn flatten<I: DoubleEndedIterator, Acc, R: Try<Ok = Acc>>(
            mut fold: impl FnMut(Acc, I::Item) -> R,
        ) -> impl FnMut(Acc, &mut I) -> R {
            move |acc, iter| iter.try_rfold(acc, &mut fold)
        }

        self.iter_try_rfold(init, flatten(fold))
    }

    #[inline]
    fn rfold<Acc, Fold>(self, init: Acc, ref mut fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        #[inline]
        fn flatten<U: DoubleEndedIterator, Acc>(
            fold: &mut impl FnMut(Acc, U::Item) -> Acc,
        ) -> impl FnMut(Acc, U) -> Acc + '_ {
            move |acc, iter| iter.rfold(acc, &mut *fold)
        }

        self.frontiter
            .into_iter()
            .chain(self.iter.map(IntoIterator::into_iter))
            .chain(self.backiter)
            .rfold(init, flatten(fold))
    }
}
